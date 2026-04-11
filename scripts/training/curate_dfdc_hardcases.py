from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PredictionRow:
    path: Path
    truth_label: str
    predicted_label: str
    decision_fake_score: float
    decision_threshold: float
    is_correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build curated DFDC dataset by label-noise filtering and hard-case upsampling.",
    )
    parser.add_argument(
        "--dataset-root",
        default="Datasets/DFDC",
        help="Original DFDC dataset root containing Fake/ and Real/ folders.",
    )
    parser.add_argument(
        "--predictions-csv",
        default="reports/domain_generalization/DFDC_sample300_dg_round1_override049/predictions.csv",
        help="Benchmark prediction csv used for hard/noise mining.",
    )
    parser.add_argument(
        "--output-root",
        default="Datasets/DFDC_Curated",
        help="Curated output dataset root.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/domain_generalization/dfdc_curation_report.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--noise-low",
        type=float,
        default=0.08,
        help="If truth=FAKE and score<=noise_low, mark as suspected noisy label.",
    )
    parser.add_argument(
        "--noise-high",
        type=float,
        default=0.92,
        help="If truth=REAL and score>=noise_high, mark as suspected noisy label.",
    )
    parser.add_argument(
        "--hard-margin",
        type=float,
        default=0.12,
        help="Near-boundary margin for hard-case selection: |score-threshold|<=hard_margin.",
    )
    parser.add_argument(
        "--hard-repeat",
        type=int,
        default=2,
        help="Extra repeats added for each selected hard sample.",
    )
    parser.add_argument(
        "--hard-max-per-class",
        type=int,
        default=120,
        help="Maximum number of hard samples selected for each class.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("link", "copy"),
        default="link",
        help="Use hard link (default) or file copy while materializing curated dataset.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear output_root before writing.",
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


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_predictions(csv_path: Path) -> List[PredictionRow]:
    rows: List[PredictionRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            raw_path = str(item.get("path", "")).strip()
            if not raw_path:
                continue
            rows.append(
                PredictionRow(
                    path=Path(raw_path),
                    truth_label=str(item.get("truth_label", "")).strip().upper(),
                    predicted_label=str(item.get("predicted_label", "")).strip().upper(),
                    decision_fake_score=to_float(item.get("decision_fake_score", 0.0)),
                    decision_threshold=to_float(item.get("decision_threshold", 0.5), 0.5),
                    is_correct=str(item.get("is_correct", "")).strip().lower() == "true",
                )
            )
    return rows


def is_noise_candidate(row: PredictionRow, noise_low: float, noise_high: float) -> bool:
    if row.truth_label == "FAKE" and row.decision_fake_score <= noise_low:
        return True
    if row.truth_label == "REAL" and row.decision_fake_score >= noise_high:
        return True
    return False


def hard_priority(row: PredictionRow) -> tuple[int, float]:
    margin = abs(row.decision_fake_score - row.decision_threshold)
    wrong = 0 if not row.is_correct else 1
    return wrong, margin


def select_hard_cases(
    rows: List[PredictionRow],
    *,
    hard_margin: float,
    hard_max_per_class: int,
    excluded_paths: set[Path],
) -> Dict[str, List[PredictionRow]]:
    buckets: Dict[str, List[PredictionRow]] = {"FAKE": [], "REAL": []}
    for row in rows:
        if row.path in excluded_paths:
            continue
        if row.truth_label not in buckets:
            continue
        near_boundary = abs(row.decision_fake_score - row.decision_threshold) <= hard_margin
        if not row.is_correct or near_boundary:
            buckets[row.truth_label].append(row)

    selected: Dict[str, List[PredictionRow]] = {"FAKE": [], "REAL": []}
    for label in ("FAKE", "REAL"):
        ordered = sorted(buckets[label], key=hard_priority)
        selected[label] = ordered[: max(0, hard_max_per_class)]
    return selected


def build_source_index(dataset_root: Path) -> Dict[Path, Path]:
    index: Dict[Path, Path] = {}
    for class_name in ("Fake", "Real"):
        folder = dataset_root / class_name
        if not folder.exists():
            continue
        for file_path in folder.iterdir():
            if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = file_path.relative_to(dataset_root)
            index[rel] = file_path
    return index


def row_to_relative(row: PredictionRow, dataset_root: Path) -> Optional[Path]:
    try:
        path = row.path.resolve()
    except FileNotFoundError:
        path = row.path
    try:
        return path.relative_to(dataset_root.resolve())
    except Exception:
        suffix = row.path.name
        class_dir = "Fake" if row.truth_label == "FAKE" else "Real"
        return Path(class_dir) / suffix


def materialize_file(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "link":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    dataset_root = resolve_project_path(args.dataset_root)
    predictions_csv = resolve_project_path(args.predictions_csv)
    output_root = resolve_project_path(args.output_root)
    report_path = resolve_project_path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "Fake").mkdir(parents=True, exist_ok=True)
    (output_root / "Real").mkdir(parents=True, exist_ok=True)

    rows = load_predictions(predictions_csv)
    source_index = build_source_index(dataset_root)

    noise_rows: List[PredictionRow] = []
    noise_rel_paths: set[Path] = set()
    usable_rows: List[PredictionRow] = []
    for row in rows:
        rel = row_to_relative(row, dataset_root)
        if rel is None:
            continue
        if rel not in source_index:
            continue
        if is_noise_candidate(row, args.noise_low, args.noise_high):
            noise_rows.append(row)
            noise_rel_paths.add(rel)
            continue
        usable_rows.append(row)

    hard_cases = select_hard_cases(
        usable_rows,
        hard_margin=float(args.hard_margin),
        hard_max_per_class=int(args.hard_max_per_class),
        excluded_paths={row.path for row in noise_rows},
    )

    copied_base = 0
    repeated_hard = 0
    hard_rel_paths: Dict[str, set[Path]] = {"FAKE": set(), "REAL": set()}
    for label in ("FAKE", "REAL"):
        for row in hard_cases[label]:
            rel = row_to_relative(row, dataset_root)
            if rel is not None:
                hard_rel_paths[label].add(rel)

    for rel, src in source_index.items():
        if rel in noise_rel_paths:
            continue
        dst = output_root / rel
        materialize_file(src, dst, args.copy_mode)
        copied_base += 1

    for label in ("FAKE", "REAL"):
        repeats = max(0, int(args.hard_repeat))
        for rel in sorted(hard_rel_paths[label]):
            src = source_index.get(rel)
            if src is None:
                continue
            class_folder = "Fake" if label == "FAKE" else "Real"
            stem = src.stem
            suffix = src.suffix
            for idx in range(repeats):
                name = f"{stem}__hard_{idx:02d}{suffix}"
                dst = output_root / class_folder / name
                materialize_file(src, dst, args.copy_mode)
                repeated_hard += 1

    fake_count = len(list((output_root / "Fake").iterdir()))
    real_count = len(list((output_root / "Real").iterdir()))

    report = {
        "dataset_root": str(dataset_root),
        "predictions_csv": str(predictions_csv),
        "output_root": str(output_root),
        "counts": {
            "input_rows": len(rows),
            "noise_candidates": len(noise_rows),
            "hard_cases_fake": len(hard_cases["FAKE"]),
            "hard_cases_real": len(hard_cases["REAL"]),
            "copied_base": copied_base,
            "repeated_hard": repeated_hard,
            "curated_fake_total": fake_count,
            "curated_real_total": real_count,
        },
        "params": {
            "noise_low": args.noise_low,
            "noise_high": args.noise_high,
            "hard_margin": args.hard_margin,
            "hard_repeat": args.hard_repeat,
            "hard_max_per_class": args.hard_max_per_class,
            "copy_mode": args.copy_mode,
        },
        "noise_examples": [
            {
                "path": str(row.path),
                "truth_label": row.truth_label,
                "predicted_label": row.predicted_label,
                "decision_fake_score": row.decision_fake_score,
                "decision_threshold": row.decision_threshold,
            }
            for row in noise_rows[:80]
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
