from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_config import get_detector_config, get_weight_path
from detectors.vision_backbones import build_vision_backbone

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SampleSplit:
    train: List[Path]
    val: List[Path]
    bench: List[Path]


class FolderDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the calibrated vision detector while holding out the benchmark sample.",
    )
    parser.add_argument("--dataset-root", default="Datasets/Test")
    parser.add_argument("--benchmark-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=3e-3)
    parser.add_argument("--lr-backbone", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report-path", default="reports/calibrated_vision_training.json")
    parser.add_argument("--disable-tta", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    project_relative = (PROJECT_ROOT / path).resolve()
    if project_relative.exists():
        return project_relative
    cwd_relative = path.resolve()
    if cwd_relative.exists():
        return cwd_relative
    return project_relative


def list_images(folder: Path) -> List[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def build_splits(dataset_root: Path, benchmark_per_class: int, seed: int, val_ratio: float) -> dict[str, SampleSplit]:
    rng = random.Random(seed)
    result: dict[str, SampleSplit] = {}
    for label, folder_name in (("FAKE", "Fake"), ("REAL", "Real")):
        files = list_images(dataset_root / folder_name)
        benchmark = set(rng.sample(files, min(benchmark_per_class, len(files))))
        remaining = [path for path in files if path not in benchmark]
        rng.shuffle(remaining)
        val_count = max(1, int(len(remaining) * val_ratio))
        result[label] = SampleSplit(
            train=remaining[:-val_count],
            val=remaining[-val_count:],
            bench=sorted(benchmark),
        )
    return result


def pack_samples(split_map: dict[str, SampleSplit], split_name: str, seed: int) -> List[Tuple[Path, int]]:
    rng = random.Random(seed)
    packed: List[Tuple[Path, int]] = []
    for label in ("FAKE", "REAL"):
        target = 1 if label == "FAKE" else 0
        packed.extend((path, target) for path in getattr(split_map[label], split_name))
    rng.shuffle(packed)
    return packed


def predict(model, loader: DataLoader, device: str, tta_enabled: bool) -> tuple[list[float], list[float]]:
    model.eval()
    probs_all: list[float] = []
    labels_all: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images).view(-1)
            if tta_enabled:
                logits = 0.5 * (logits + model(torch.flip(images, dims=[3])).view(-1))
            probs = torch.sigmoid(logits)
            probs_all.extend(probs.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())
    return probs_all, labels_all


def evaluate(probs: Iterable[float], labels: Iterable[float], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    probs_list = list(probs)
    labels_list = list(labels)
    for prob, label in zip(probs_list, labels_list):
        pred = 1.0 if prob >= threshold else 0.0
        if pred == 1.0 and label == 1.0:
            tp += 1
        elif pred == 0.0 and label == 0.0:
            tn += 1
        elif pred == 1.0 and label == 0.0:
            fp += 1
        else:
            fn += 1
    total = max(len(labels_list), 1)
    accuracy = (tp + tn) / total
    balanced_accuracy = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(probs: Sequence[float], labels: Sequence[float]) -> tuple[float, dict]:
    best = None
    for threshold in [idx / 100 for idx in range(15, 86)]:
        metrics = evaluate(probs, labels, threshold)
        score = (metrics["accuracy"], metrics["balanced_accuracy"], metrics["tp"] - metrics["fp"])
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    assert best is not None
    return best[1], best[2]


def configure_optimizer(model, *, head_lr: float, backbone_lr: float, weight_decay: float):
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(key in name for key in ("classifier", "fc", "head")):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    dataset_root = resolve_project_path(args.dataset_root)
    report_path = resolve_project_path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    config = get_detector_config("CalibratedVision")
    runtime = config.runtime_params
    variant = str(runtime.get("variant", "efficientnet_b0_ft_v1"))
    output_path = resolve_project_path(args.output) if args.output else get_weight_path("CalibratedVision")
    if output_path is None:
        raise RuntimeError("CalibratedVision weight path is not configured")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from torchvision import transforms

    resize_size = int(runtime.get("resize_size", 256))
    input_size = int(runtime.get("input_size", 224))
    mean = runtime.get("mean", [0.485, 0.456, 0.406])
    std = runtime.get("std", [0.229, 0.224, 0.225])
    dropout = float(runtime.get("dropout", 0.2))

    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(input_size, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
            transforms.RandomAutocontrast(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    splits = build_splits(dataset_root, args.benchmark_per_class, args.seed, args.val_ratio)
    train_samples = pack_samples(splits, "train", args.seed)
    val_samples = pack_samples(splits, "val", args.seed + 1)
    bench_samples = pack_samples(splits, "bench", args.seed + 2)

    train_loader = DataLoader(
        FolderDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        FolderDataset(val_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    bench_loader = DataLoader(
        FolderDataset(bench_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = build_vision_backbone(
        variant,
        pretrained=True,
        dropout=dropout,
    ).to(device)

    if variant == "efficientnet_b0_ft_v1":
        for name, parameter in model.named_parameters():
            if "classifier" not in name:
                parameter.requires_grad = False

    optimizer = configure_optimizer(
        model,
        head_lr=args.lr_head,
        backbone_lr=args.lr_backbone,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.15], device=device))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    tta_enabled = not args.disable_tta

    best_state = None
    best_threshold = config.decision_threshold
    best_val_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        if variant == "efficientnet_b0_ft_v1" and epoch == args.warmup_epochs + 1:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = configure_optimizer(
                model,
                head_lr=args.lr_head,
                backbone_lr=args.lr_backbone,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs - args.warmup_epochs, 1),
            )

        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(images).view(-1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
        scheduler.step()

        val_probs, val_labels = predict(model, val_loader, device, tta_enabled)
        epoch_threshold, epoch_metrics = find_best_threshold(val_probs, val_labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(len(train_loader.dataset), 1),
                "val_best_threshold": epoch_threshold,
                "val_metrics": epoch_metrics,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False))

        if best_val_metrics is None or epoch_metrics["accuracy"] > best_val_metrics["accuracy"]:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_threshold = epoch_threshold
            best_val_metrics = epoch_metrics

    assert best_state is not None and best_val_metrics is not None
    model.load_state_dict(best_state)

    val_probs, val_labels = predict(model, val_loader, device, tta_enabled)
    bench_probs, bench_labels = predict(model, bench_loader, device, tta_enabled)
    final_val_metrics = evaluate(val_probs, val_labels, best_threshold)
    final_bench_metrics = evaluate(bench_probs, bench_labels, best_threshold)

    checkpoint = {
        "state_dict": best_state,
        "meta": {
            "training_version": variant,
            "decision_threshold": best_threshold,
            "resize_size": resize_size,
            "input_size": input_size,
            "mean": list(mean),
            "std": list(std),
            "dropout": dropout,
            "seed": args.seed,
            "benchmark_per_class": args.benchmark_per_class,
            "tta_enabled": tta_enabled,
        },
    }
    torch.save(checkpoint, output_path)

    report = {
        "dataset_root": str(dataset_root),
        "output_path": str(output_path),
        "history": history,
        "best_threshold": best_threshold,
        "val_metrics": final_val_metrics,
        "bench_metrics": final_bench_metrics,
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "bench": len(bench_samples),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
