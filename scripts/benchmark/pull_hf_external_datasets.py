from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
REPO_CELEB = "RohanRamesh/celebdfv2_224"
REPO_DFDC = "mkhLlamaLearn/dfdcpics2"
REPO_WILD = "xingjunm/WildDeepfake"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull Celeb-DF/DFDC/WildDeepfake subsets from Hugging Face.")
    parser.add_argument("--dataset-root", default="Datasets", help="Local dataset root.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Celeb-DF", "DFDC", "WildDeepfake"],
        choices=["Celeb-DF", "DFDC", "WildDeepfake"],
        help="Which datasets to pull.",
    )
    parser.add_argument("--per-class", type=int, default=300, help="Target images per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing Fake/Real files before pulling (recommended for reproducibility).",
    )
    return parser.parse_args()


def ensure_class_dir(root: Path, cls: str, clear_existing: bool) -> Path:
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for p in d.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    return d


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def write_manifest(root: Path, payload: dict) -> None:
    (root / "_sample_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def pull_celebdf(dataset_root: Path, per_class: int, seed: int, clear_existing: bool, api: HfApi) -> None:
    name = "Celeb-DF"
    root = dataset_root / name
    fake_dir = ensure_class_dir(root, "Fake", clear_existing)
    real_dir = ensure_class_dir(root, "Real", clear_existing)
    rng = random.Random(seed)

    targets = {
        "Fake": ("test/fake", fake_dir),
        "Real": ("test/real", real_dir),
    }
    counts = {"Fake": count_images(fake_dir), "Real": count_images(real_dir)}

    for cls, (prefix, out_dir) in targets.items():
        need = max(per_class - counts[cls], 0)
        if need == 0:
            continue

        entries = list(
            api.list_repo_tree(REPO_CELEB, repo_type="dataset", path_in_repo=prefix, recursive=True)
        )
        files = [e.path for e in entries if str(getattr(e, "path", "")).lower().endswith(IMAGE_SUFFIXES)]
        if not files:
            continue
        rng.shuffle(files)
        picks = files[:need]

        start_idx = counts[cls] + 1
        for i, fpath in enumerate(picks, start=start_idx):
            local = hf_hub_download(repo_id=REPO_CELEB, repo_type="dataset", filename=fpath)
            ext = Path(fpath).suffix.lower() or ".jpg"
            out_name = f"celebdf_{cls.lower()}_{i:05d}{ext}"
            shutil.copy2(local, out_dir / out_name)
        counts[cls] = count_images(out_dir)

    write_manifest(
        root,
        {
            "dataset": name,
            "repo": REPO_CELEB,
            "seed": seed,
            "target_per_class": per_class,
            "counts": counts,
            "note": "Pulled from HF image tree.",
        },
    )


def pull_dfdc(dataset_root: Path, per_class: int, seed: int, clear_existing: bool, api: HfApi) -> None:
    del seed
    name = "DFDC"
    root = dataset_root / name
    fake_dir = ensure_class_dir(root, "Fake", clear_existing)
    real_dir = ensure_class_dir(root, "Real", clear_existing)
    counts = {"Fake": count_images(fake_dir), "Real": count_images(real_dir)}

    parquet_files = sorted(
        f
        for f in api.list_repo_files(REPO_DFDC, repo_type="dataset")
        if f.startswith("data/train-") and f.endswith(".parquet")
    )

    for pf in parquet_files:
        if counts["Fake"] >= per_class and counts["Real"] >= per_class:
            break

        local_pf = hf_hub_download(repo_id=REPO_DFDC, repo_type="dataset", filename=pf)
        table = pq.read_table(local_pf, columns=["image", "label"])
        data = table.to_pydict()
        images = data.get("image", [])
        labels = data.get("label", [])

        for image_item, label_item in zip(images, labels):
            label = str(label_item).strip().upper()
            if label not in {"FAKE", "REAL"}:
                continue
            cls = "Fake" if label == "FAKE" else "Real"
            if counts[cls] >= per_class:
                continue
            if not isinstance(image_item, dict):
                continue
            image_bytes = image_item.get("bytes")
            image_path = str(image_item.get("path") or "")
            if not image_bytes:
                continue

            ext = Path(image_path).suffix.lower() if image_path else ".jpg"
            if ext not in IMAGE_SUFFIXES:
                ext = ".jpg"
            idx = counts[cls] + 1
            out_name = f"dfdc_{cls.lower()}_{idx:05d}{ext}"
            out_dir = fake_dir if cls == "Fake" else real_dir
            (out_dir / out_name).write_bytes(image_bytes)
            counts[cls] += 1

    write_manifest(
        root,
        {
            "dataset": name,
            "repo": REPO_DFDC,
            "target_per_class": per_class,
            "counts": counts,
            "note": "Extracted from parquet(image bytes + label).",
        },
    )


def _extract_from_wild_shards(
    api: HfApi,
    prefix: str,
    out_dir: Path,
    out_prefix: str,
    target_count: int,
    seed: int,
) -> int:
    rng = random.Random(seed)
    current = count_images(out_dir)
    if current >= target_count:
        return current

    entries = list(
        api.list_repo_tree(REPO_WILD, repo_type="dataset", path_in_repo=prefix, recursive=False, expand=True)
    )
    shards = []
    for it in entries:
        p = str(getattr(it, "path", ""))
        if p.endswith(".tar.gz"):
            size = int(getattr(it, "size", 0) or 0)
            shards.append((p, size))
    shards.sort(key=lambda x: x[1])

    for shard_path, _ in shards:
        if current >= target_count:
            break
        local_tar = hf_hub_download(repo_id=REPO_WILD, repo_type="dataset", filename=shard_path)
        try:
            tf = tarfile.open(local_tar, mode="r:*")
        except Exception:
            continue

        with tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(IMAGE_SUFFIXES)]
            if not members:
                continue
            rng.shuffle(members)
            for m in members:
                if current >= target_count:
                    break
                fo = tf.extractfile(m)
                if fo is None:
                    continue
                data = fo.read()
                if not data:
                    continue
                ext = Path(m.name).suffix.lower() or ".jpg"
                if ext not in IMAGE_SUFFIXES:
                    ext = ".jpg"
                current += 1
                out_name = f"{out_prefix}_{current:05d}{ext}"
                (out_dir / out_name).write_bytes(data)

    return current


def pull_wild(dataset_root: Path, per_class: int, seed: int, clear_existing: bool, api: HfApi) -> None:
    name = "WildDeepfake"
    root = dataset_root / name
    fake_dir = ensure_class_dir(root, "Fake", clear_existing)
    real_dir = ensure_class_dir(root, "Real", clear_existing)

    fake_count = _extract_from_wild_shards(
        api=api,
        prefix="deepfake_in_the_wild/fake_test",
        out_dir=fake_dir,
        out_prefix="wild_fake",
        target_count=per_class,
        seed=seed,
    )
    real_count = _extract_from_wild_shards(
        api=api,
        prefix="deepfake_in_the_wild/real_test",
        out_dir=real_dir,
        out_prefix="wild_real",
        target_count=per_class,
        seed=seed + 1,
    )

    write_manifest(
        root,
        {
            "dataset": name,
            "repo": REPO_WILD,
            "seed": seed,
            "target_per_class": per_class,
            "counts": {"Fake": fake_count, "Real": real_count},
            "note": "Extracted from fake_test/real_test tar shards.",
        },
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    if "Celeb-DF" in args.datasets:
        pull_celebdf(dataset_root, args.per_class, args.seed, args.clear_existing, api)
    if "DFDC" in args.datasets:
        pull_dfdc(dataset_root, args.per_class, args.seed, args.clear_existing, api)
    if "WildDeepfake" in args.datasets:
        pull_wild(dataset_root, args.per_class, args.seed, args.clear_existing, api)

    print("DONE")
    for ds in args.datasets:
        for cls in ("Fake", "Real"):
            d = dataset_root / ds / cls
            n = count_images(d) if d.exists() else 0
            print(f"{ds}/{cls}: {n}")


if __name__ == "__main__":
    main()
