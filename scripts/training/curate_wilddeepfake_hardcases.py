from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.curate_dfdc_hardcases import (
    PredictionRow,
    build_source_index,
    is_noise_candidate,
    load_predictions,
    materialize_file,
    resolve_project_path,
    row_to_relative,
    select_hard_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build curated WildDeepfake dataset by label-noise filtering and hard-case upsampling.",
    )
    parser.add_argument(
        "--dataset-root",
        default="Datasets/WildDeepfake",
        help="Original WildDeepfake dataset root containing Fake/ and Real/ folders.",
    )
    parser.add_argument(
        "--predictions-csv",
        default="reports/report_wilddeepfake_sample1200_override_010_2026-04-20/predictions.csv",
        help="Benchmark prediction csv used for hard/noise mining.",
    )
    parser.add_argument(
        "--output-root",
        default="Datasets/WildDeepfake_Curated",
        help="Curated output dataset root.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/report_wilddeepfake_curation_report.json",
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
        default=0.08,
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
        default=180,
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

    noise_rows = []
    noise_rel_paths = set()
    usable_rows = []
    for row in rows:
        rel = row_to_relative(row, dataset_root)
        if rel is None or rel not in source_index:
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
    hard_rel_paths = {"FAKE": set(), "REAL": set()}
    for label in ("FAKE", "REAL"):
        for row in hard_cases[label]:
            rel = row_to_relative(row, dataset_root)
            if rel is not None:
                hard_rel_paths[label].add(rel)

    for rel, src in source_index.items():
        if rel in noise_rel_paths:
            continue
        materialize_file(src, output_root / rel, args.copy_mode)
        copied_base += 1

    for label in ("FAKE", "REAL"):
        repeats = max(0, int(args.hard_repeat))
        class_folder = "Fake" if label == "FAKE" else "Real"
        for rel in sorted(hard_rel_paths[label]):
            src = source_index.get(rel)
            if src is None:
                continue
            for idx in range(repeats):
                dst = output_root / class_folder / f"{src.stem}__hard_{idx:02d}{src.suffix}"
                materialize_file(src, dst, args.copy_mode)
                repeated_hard += 1

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
            "curated_fake_total": len(list((output_root / "Fake").iterdir())),
            "curated_real_total": len(list((output_root / "Real").iterdir())),
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
