from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
class LabeledSample:
    path: Path
    label: int
    dataset: str


class MixedDomainDataset(Dataset):
    def __init__(self, samples: Sequence[LabeledSample], transform, dataset_to_id: Dict[str, int]):
        self.samples = list(samples)
        self.transform = transform
        self.dataset_to_id = dict(dataset_to_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(sample.label, dtype=torch.float32),
            torch.tensor(self.dataset_to_id[sample.dataset], dtype=torch.int64),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a domain-generalized CalibratedVision detector on multiple datasets.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=[
            "Datasets/Test",
            "Datasets/Celeb-DF",
            "Datasets/DFDC",
            "Datasets/WildDeepfake",
        ],
        help="Dataset roots containing Fake/ and Real/ subfolders.",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=0,
        help="Optional cap on train samples per class for each dataset. 0 means no cap.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=2e-3)
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint (.pt/.pth) to initialize model weights before training.",
    )
    parser.add_argument("--output", default="weights/calibrated_vision_detector_dg.pt")
    parser.add_argument(
        "--report-path",
        default="reports/report_training_calibrated_vision_dg.json",
    )
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


def split_dataset(
    dataset_name: str,
    dataset_root: Path,
    *,
    val_ratio: float,
    seed: int,
    train_per_class: int,
) -> tuple[List[LabeledSample], List[LabeledSample]]:
    rng = random.Random(seed)
    train_samples: List[LabeledSample] = []
    val_samples: List[LabeledSample] = []

    for class_name, label in (("Fake", 1), ("Real", 0)):
        files = list_images(dataset_root / class_name)
        rng.shuffle(files)
        val_count = max(1, int(len(files) * val_ratio))
        val_part = files[:val_count]
        train_part = files[val_count:]
        if train_per_class > 0 and len(train_part) > train_per_class:
            train_part = train_part[:train_per_class]

        train_samples.extend(
            LabeledSample(path=path, label=label, dataset=dataset_name)
            for path in train_part
        )
        val_samples.extend(
            LabeledSample(path=path, label=label, dataset=dataset_name)
            for path in val_part
        )

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


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


def predict(
    model,
    loader: DataLoader,
    device: str,
    *,
    tta_enabled: bool,
) -> tuple[list[float], list[int], list[int]]:
    model.eval()
    probs_all: list[float] = []
    labels_all: list[int] = []
    dataset_ids_all: list[int] = []
    with torch.no_grad():
        for images, labels, dataset_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images).view(-1)
            if tta_enabled:
                logits = 0.5 * (logits + model(torch.flip(images, dims=[3])).view(-1))
            probs = torch.sigmoid(logits)
            probs_all.extend(probs.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().to(torch.int64).tolist())
            dataset_ids_all.extend(dataset_ids.detach().cpu().tolist())
    return probs_all, labels_all, dataset_ids_all


def evaluate_binary(probs: Iterable[float], labels: Iterable[int], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    probs_list = list(probs)
    labels_list = list(labels)
    for prob, label in zip(probs_list, labels_list):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    total = max(len(labels_list), 1)
    accuracy = (tp + tn) / total
    recall_fake = tp / max(tp + fn, 1)
    specificity_real = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (recall_fake + specificity_real)
    precision_fake = tp / max(tp + fp, 1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "specificity_real": specificity_real,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(probs: Sequence[float], labels: Sequence[int]) -> tuple[float, dict]:
    best = None
    for threshold in [idx / 100 for idx in range(10, 91)]:
        metrics = evaluate_binary(probs, labels, threshold)
        score = (
            metrics["balanced_accuracy"],
            metrics["accuracy"],
            metrics["tp"] - metrics["fp"],
        )
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    assert best is not None
    return float(best[1]), dict(best[2])


def find_best_threshold_per_dataset(
    probs: Sequence[float],
    labels: Sequence[int],
    dataset_ids: Sequence[int],
    dataset_names: Sequence[str],
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for ds_id, ds_name in enumerate(dataset_names):
        filtered = [
            (p, y)
            for p, y, d in zip(probs, labels, dataset_ids)
            if int(d) == ds_id
        ]
        if not filtered:
            continue
        ds_probs = [item[0] for item in filtered]
        ds_labels = [item[1] for item in filtered]
        threshold, _ = find_best_threshold(ds_probs, ds_labels)
        output[ds_name] = threshold
    return output


def average_balanced_accuracy(per_dataset: Dict[str, dict], domains: Sequence[str]) -> float:
    values = [
        float(per_dataset[domain]["balanced_accuracy"])
        for domain in domains
        if domain in per_dataset
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_guardrail_gap(
    payload: dict,
    *,
    guardrail_domains: Sequence[str],
    guardrail_min_balanced_accuracy: float,
) -> float:
    per_dataset = payload.get("per_dataset", {})
    gap = 0.0
    for domain in guardrail_domains:
        score = float(per_dataset.get(domain, {}).get("balanced_accuracy", 0.0))
        gap += max(float(guardrail_min_balanced_accuracy) - score, 0.0)
    return gap


def summarize_epoch_selection(
    payload: dict,
    *,
    guardrail_domains: Sequence[str],
    target_domains: Sequence[str],
    guardrail_min_balanced_accuracy: float,
) -> dict:
    per_dataset = payload.get("per_dataset", {})
    return {
        "epoch": int(payload.get("epoch", 0)),
        "guardrail_domains": [str(domain) for domain in guardrail_domains],
        "target_domains": [str(domain) for domain in target_domains],
        "guardrail_min_balanced_accuracy": round(float(guardrail_min_balanced_accuracy), 6),
        "guardrail_average": round(average_balanced_accuracy(per_dataset, guardrail_domains), 6),
        "target_average": round(average_balanced_accuracy(per_dataset, target_domains), 6),
        "guardrail_gap": round(
            compute_guardrail_gap(
                payload,
                guardrail_domains=guardrail_domains,
                guardrail_min_balanced_accuracy=guardrail_min_balanced_accuracy,
            ),
            6,
        ),
    }


def choose_best_epoch_payload(
    payloads: Sequence[dict],
    *,
    guardrail_domains: Sequence[str],
    guardrail_min_balanced_accuracy: float,
    target_domains: Sequence[str],
) -> dict:
    if not payloads:
        raise ValueError("payloads must not be empty")

    def with_summary(payload: dict) -> dict:
        summary = summarize_epoch_selection(
            payload,
            guardrail_domains=guardrail_domains,
            target_domains=target_domains,
            guardrail_min_balanced_accuracy=guardrail_min_balanced_accuracy,
        )
        enriched = dict(payload)
        enriched["selection_summary"] = summary
        return enriched

    enriched_payloads = [with_summary(payload) for payload in payloads]
    qualified = [
        payload
        for payload in enriched_payloads
        if float(payload["selection_summary"]["guardrail_gap"]) <= 1e-12
    ]
    candidates = qualified if qualified else enriched_payloads

    def score(payload: dict) -> tuple:
        summary = payload["selection_summary"]
        val_metrics = payload.get("val_metrics", {})
        if qualified:
            return (
                float(summary["target_average"]),
                float(payload.get("mean_dataset_balanced_accuracy", 0.0)),
                float(val_metrics.get("balanced_accuracy", 0.0)),
                float(val_metrics.get("accuracy", 0.0)),
            )
        return (
            -float(summary["guardrail_gap"]),
            float(summary["target_average"]),
            float(payload.get("mean_dataset_balanced_accuracy", 0.0)),
            float(val_metrics.get("balanced_accuracy", 0.0)),
        )

    return max(candidates, key=score)


def evaluate_by_dataset(
    probs: Sequence[float],
    labels: Sequence[int],
    dataset_ids: Sequence[int],
    dataset_names: Sequence[str],
    threshold: float,
) -> Dict[str, dict]:
    output: Dict[str, dict] = {}
    for ds_id, ds_name in enumerate(dataset_names):
        filtered = [
            (p, y)
            for p, y, d in zip(probs, labels, dataset_ids)
            if int(d) == ds_id
        ]
        if not filtered:
            continue
        ds_probs = [item[0] for item in filtered]
        ds_labels = [item[1] for item in filtered]
        metrics = evaluate_binary(ds_probs, ds_labels, threshold)
        metrics["count"] = len(ds_probs)
        output[ds_name] = metrics
    return output


def resolve_guardrail_domains(dataset_names: Sequence[str]) -> List[str]:
    preferred = ["Test", "Celeb-DF"]
    return [name for name in preferred if name in dataset_names]


def resolve_target_domains(dataset_names: Sequence[str]) -> List[str]:
    preferred = ["DFDC_Curated", "WildDeepfake_Curated", "DFDC", "WildDeepfake"]
    output = [name for name in preferred if name in dataset_names]
    if output:
        return output
    return [name for name in dataset_names if "dfdc" in name.lower() or "wild" in name.lower()]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    config = get_detector_config("CalibratedVision")
    runtime = config.runtime_params
    variant = str(runtime.get("variant", "efficientnet_b0_ft_v1"))
    resize_size = int(runtime.get("resize_size", 256))
    input_size = int(runtime.get("input_size", 224))
    mean = runtime.get("mean", [0.485, 0.456, 0.406])
    std = runtime.get("std", [0.229, 0.224, 0.225])
    dropout = float(runtime.get("dropout", 0.2))

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = resolve_project_path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(input_size, scale=(0.60, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03)],
                p=0.75,
            ),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomGrayscale(p=0.05),
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

    dataset_roots = [resolve_project_path(path) for path in args.dataset_roots]
    dataset_names = [path.name for path in dataset_roots]
    dataset_to_id = {name: idx for idx, name in enumerate(dataset_names)}
    guardrail_domains = resolve_guardrail_domains(dataset_names)
    target_domains = resolve_target_domains(dataset_names)
    guardrail_min_balanced_accuracy = 0.90

    all_train: List[LabeledSample] = []
    all_val: List[LabeledSample] = []
    split_stats: Dict[str, Dict[str, int]] = {}
    for index, dataset_root in enumerate(dataset_roots):
        dataset_name = dataset_names[index]
        train_samples, val_samples = split_dataset(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed) + index,
            train_per_class=int(args.train_per_class),
        )
        all_train.extend(train_samples)
        all_val.extend(val_samples)
        split_stats[dataset_name] = {
            "train_count": len(train_samples),
            "val_count": len(val_samples),
        }

    random.Random(args.seed).shuffle(all_train)
    random.Random(args.seed + 1).shuffle(all_val)

    train_loader = DataLoader(
        MixedDomainDataset(all_train, train_transform, dataset_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        MixedDomainDataset(all_val, eval_transform, dataset_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_vision_backbone(
        variant,
        pretrained=True,
        dropout=dropout,
    ).to(device)

    if args.init_checkpoint:
        init_path = resolve_project_path(args.init_checkpoint)
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        print(f"[RELOAD] loaded init checkpoint: {init_path}")

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
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.10], device=device))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    tta_enabled = not args.disable_tta

    best_state = None
    best_threshold = float(config.decision_threshold)
    best_payload = None
    history: List[dict] = []

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
        for images, labels, _ in train_loader:
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

        val_probs, val_labels, val_dataset_ids = predict(
            model,
            val_loader,
            device,
            tta_enabled=tta_enabled,
        )
        epoch_threshold, epoch_metrics = find_best_threshold(val_probs, val_labels)
        per_dataset = evaluate_by_dataset(
            val_probs,
            val_labels,
            val_dataset_ids,
            dataset_names,
            epoch_threshold,
        )
        mean_ds_bal_acc = (
            sum(item["balanced_accuracy"] for item in per_dataset.values()) / max(len(per_dataset), 1)
        )

        epoch_payload = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader.dataset), 1),
            "threshold": epoch_threshold,
            "val_metrics": epoch_metrics,
            "mean_dataset_balanced_accuracy": mean_ds_bal_acc,
            "per_dataset": per_dataset,
        }
        history.append(epoch_payload)
        print(json.dumps(epoch_payload, ensure_ascii=False))

        candidate_payload = choose_best_epoch_payload(
            [epoch_payload] if best_payload is None else [best_payload, epoch_payload],
            guardrail_domains=guardrail_domains,
            guardrail_min_balanced_accuracy=guardrail_min_balanced_accuracy,
            target_domains=target_domains,
        )
        if candidate_payload.get("epoch") == epoch:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_threshold = epoch_threshold
            best_payload = candidate_payload

    assert best_state is not None and best_payload is not None
    model.load_state_dict(best_state)

    val_probs, val_labels, val_dataset_ids = predict(
        model,
        val_loader,
        device,
        tta_enabled=tta_enabled,
    )
    final_metrics = evaluate_binary(val_probs, val_labels, best_threshold)
    final_per_dataset = evaluate_by_dataset(
        val_probs,
        val_labels,
        val_dataset_ids,
        dataset_names,
        best_threshold,
    )
    per_domain_thresholds = find_best_threshold_per_dataset(
        val_probs,
        val_labels,
        val_dataset_ids,
        dataset_names,
    )

    checkpoint = {
        "state_dict": best_state,
        "meta": {
            "training_version": variant,
            "decision_threshold": float(best_threshold),
            "domain_thresholds": per_domain_thresholds,
            "resize_size": resize_size,
            "input_size": input_size,
            "mean": list(mean),
            "std": list(std),
            "dropout": dropout,
            "seed": args.seed,
            "tta_enabled": tta_enabled,
            "dataset_roots": [str(path) for path in dataset_roots],
        },
    }
    torch.save(checkpoint, output_path)

    report = {
        "output_path": str(output_path),
        "best_threshold": float(best_threshold),
        "global_val_metrics": final_metrics,
        "per_dataset_val_metrics": final_per_dataset,
        "per_dataset_recommended_thresholds": per_domain_thresholds,
        "guardrail_domains": guardrail_domains,
        "target_domains": target_domains,
        "selection_summary": best_payload.get("selection_summary", {}),
        "split_stats": split_stats,
        "history": history,
        "best_epoch_payload": best_payload,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
