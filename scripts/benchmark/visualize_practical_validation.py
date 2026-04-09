from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.visualize_detect_benchmark import (
    DatasetSample,
    DetectBenchmarkRunner,
    EXCLUDED_LABELS,
    HttpDetectClient,
    IMAGE_SUFFIXES,
    InternalDetectClient,
    PredictionRecord,
    VALID_LABELS,
    collect_samples,
    compute_summary,
    format_percent,
)


RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    title: str
    description: str
    transform: Callable[[Image.Image], Image.Image]
    jpeg_quality: int = 92


@dataclass(frozen=True)
class LabelledSuiteResult:
    key: str
    title: str
    suite_kind: str
    dataset_root: str
    transform_name: str
    transform_title: str
    sample_count: int
    average_latency_ms: float
    metrics: Dict[str, float | int | str]


@dataclass(frozen=True)
class ScopeSuiteResult:
    key: str
    title: str
    dataset_root: str
    total_samples: int
    rejected_predictions: int
    accepted_predictions: int
    error_count: int
    rejection_rate: float
    acceptance_rate: float
    error_rate: float
    average_latency_ms: float
    label_breakdown: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a practical undergraduate-ready validation pack: in-domain test, "
            "cross-scene validation, robustness perturbations, and out-of-scope probing."
        ),
    )
    parser.add_argument(
        "--test-root",
        default="Datasets/Test",
        help="In-domain labelled dataset root containing Fake/ and Real/.",
    )
    parser.add_argument(
        "--validation-root",
        default="Datasets/Validation",
        help="Cross-scene labelled dataset root containing Fake/ and Real/.",
    )
    parser.add_argument(
        "--nobody-root",
        default="Datasets/Nobody",
        help="Optional out-of-scope dataset root. Files can live directly under this folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/practical_validation",
        help="Directory used to write HTML, CSV, and JSON validation outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=("internal", "http"),
        default="internal",
        help="Use internal detect facade or call a running /detect endpoint.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8001/detect",
        help="HTTP endpoint used when --mode=http.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for --mode=http.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.80,
        help="semantic_threshold forwarded to detect requests.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=80,
        help="Sample count per class for the Test and Validation original suites.",
    )
    parser.add_argument(
        "--robustness-sample-per-class",
        type=int,
        default=30,
        help="Sample count per class for each perturbation suite.",
    )
    parser.add_argument(
        "--nobody-limit",
        type=int,
        default=20,
        help="Maximum number of Nobody images used for out-of-scope probing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of benchmark worker threads. Internal mode is more stable with 1.",
    )
    parser.add_argument(
        "--skip-scope",
        action="store_true",
        help="Skip the Nobody out-of-scope probe even if the directory exists.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip JPEG, blur, and resize perturbation suites.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    raw_path = Path(path_str)
    if raw_path.is_absolute():
        return raw_path.resolve()

    project_relative = (PROJECT_ROOT / raw_path).resolve()
    if project_relative.exists():
        return project_relative

    cwd_relative = raw_path.resolve()
    if cwd_relative.exists():
        return cwd_relative

    return project_relative


def build_client(mode: str, endpoint: str, timeout: float):
    if mode == "internal":
        return InternalDetectClient()
    return HttpDetectClient(endpoint, timeout)


def jpeg_transform(image: Image.Image) -> Image.Image:
    return image.copy()


def blur_transform(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=1.2))


def resize_transform(image: Image.Image) -> Image.Image:
    width, height = image.size
    down_width = max(64, int(width * 0.60))
    down_height = max(64, int(height * 0.60))
    reduced = image.resize((down_width, down_height), RESAMPLING.BILINEAR)
    return reduced.resize((width, height), RESAMPLING.BICUBIC)


def build_perturbations() -> List[PerturbationSpec]:
    return [
        PerturbationSpec(
            name="jpeg_q60",
            title="JPEG Q60",
            description="Simulate social-platform recompression by re-encoding each image at JPEG quality 60.",
            transform=jpeg_transform,
            jpeg_quality=60,
        ),
        PerturbationSpec(
            name="gaussian_blur",
            title="Gaussian Blur",
            description="Apply a mild Gaussian blur to simulate low-quality capture or secondary reposting.",
            transform=blur_transform,
        ),
        PerturbationSpec(
            name="downscale_restore",
            title="Downscale + Restore",
            description="Shrink to 60% and resize back to original resolution to simulate screen-capture or platform scaling.",
            transform=resize_transform,
        ),
    ]


def materialize_samples(
    samples: List[DatasetSample],
    perturbation: PerturbationSpec,
    workspace: Path,
) -> List[DatasetSample]:
    generated: List[DatasetSample] = []
    target_root = workspace / perturbation.name
    target_root.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        class_dir = target_root / sample.truth_label
        class_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_dir / f"{sample.path.stem}_{perturbation.name}_{index:04d}.jpg"

        with Image.open(sample.path) as image:
            transformed = perturbation.transform(image.convert("RGB"))
            transformed.save(
                output_path,
                format="JPEG",
                quality=perturbation.jpeg_quality,
                optimize=False,
            )

        generated.append(DatasetSample(path=output_path, truth_label=sample.truth_label))
    return generated


def collect_scope_samples(dataset_root: Path, limit: Optional[int], seed: int) -> List[DatasetSample]:
    if not dataset_root.exists():
        return []

    candidates = sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        return []

    if limit is not None and len(candidates) > limit:
        import random

        rng = random.Random(seed)
        candidates = sorted(rng.sample(candidates, limit))

    return [DatasetSample(path=path, truth_label="OUT_OF_SCOPE") for path in candidates]


def average(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def run_labelled_suite(
    *,
    key: str,
    title: str,
    suite_kind: str,
    dataset_root: Path,
    sample_per_class: int,
    seed: int,
    runner: DetectBenchmarkRunner,
    perturbation: Optional[PerturbationSpec] = None,
    workspace: Optional[Path] = None,
) -> tuple[LabelledSuiteResult, List[PredictionRecord]]:
    samples = collect_samples(
        dataset_root=dataset_root,
        limit_per_class=None,
        sample_per_class=sample_per_class,
        seed=seed,
    )

    transform_name = "original"
    transform_title = "Original"
    if perturbation is not None:
        if workspace is None:
            raise RuntimeError("workspace is required for perturbation suites")
        samples = materialize_samples(samples, perturbation, workspace)
        transform_name = perturbation.name
        transform_title = perturbation.title

    records = runner.run(samples)
    summary = compute_summary(records)
    result = LabelledSuiteResult(
        key=key,
        title=title,
        suite_kind=suite_kind,
        dataset_root=str(dataset_root),
        transform_name=transform_name,
        transform_title=transform_title,
        sample_count=len(records),
        average_latency_ms=average(record.latency_ms for record in records),
        metrics={
            "total_samples": summary.total_samples,
            "excluded_predictions": summary.excluded_predictions,
            "valid_predictions": summary.valid_predictions,
            "error_count": summary.error_count,
            "correct_predictions": summary.correct_predictions,
            "accuracy_all": summary.accuracy_all,
            "accuracy_valid": summary.accuracy_valid,
            "precision_fake": summary.precision_fake,
            "recall_fake": summary.recall_fake,
            "specificity_real": summary.specificity_real,
            "f1_fake": summary.f1_fake,
            "balanced_accuracy": summary.balanced_accuracy,
            "avg_confidence_correct": summary.avg_confidence_correct,
            "avg_confidence_incorrect": summary.avg_confidence_incorrect,
            "tp": summary.tp,
            "tn": summary.tn,
            "fp": summary.fp,
            "fn": summary.fn,
            "fake_support": summary.fake_support,
            "real_support": summary.real_support,
        },
    )
    return result, records


def run_scope_suite(
    *,
    key: str,
    title: str,
    dataset_root: Path,
    limit: int,
    seed: int,
    runner: DetectBenchmarkRunner,
) -> tuple[Optional[ScopeSuiteResult], List[PredictionRecord]]:
    samples = collect_scope_samples(dataset_root, limit, seed)
    if not samples:
        return None, []

    records = runner.run(samples)
    label_breakdown: Dict[str, int] = {}
    for record in records:
        label_breakdown[record.predicted_label] = label_breakdown.get(record.predicted_label, 0) + 1

    rejected = sum(1 for record in records if record.predicted_label in EXCLUDED_LABELS)
    accepted = sum(1 for record in records if record.predicted_label in VALID_LABELS)
    errors = sum(
        1
        for record in records
        if record.predicted_label not in VALID_LABELS and record.predicted_label not in EXCLUDED_LABELS
    )
    summary = ScopeSuiteResult(
        key=key,
        title=title,
        dataset_root=str(dataset_root),
        total_samples=len(records),
        rejected_predictions=rejected,
        accepted_predictions=accepted,
        error_count=errors,
        rejection_rate=safe_div(rejected, len(records)),
        acceptance_rate=safe_div(accepted, len(records)),
        error_rate=safe_div(errors, len(records)),
        average_latency_ms=average(record.latency_ms for record in records),
        label_breakdown=dict(sorted(label_breakdown.items())),
    )
    return summary, records


def score_metric(value: float, strong_threshold: float, pass_threshold: float) -> float:
    if value >= strong_threshold:
        return 1.0
    if value >= pass_threshold:
        return 0.7
    return 0.35


def build_overall_summary(
    labelled_results: List[LabelledSuiteResult],
    scope_result: Optional[ScopeSuiteResult],
) -> Dict[str, float | int | str]:
    metrics_by_key = {item.key: item for item in labelled_results}
    in_domain = float(metrics_by_key["test_in_domain"].metrics["accuracy_valid"])
    cross_scene = float(metrics_by_key["validation_cross_scene"].metrics["accuracy_valid"])
    robustness_items = [item for item in labelled_results if item.suite_kind == "robustness"]
    robustness_accuracy = (
        average(float(item.metrics["accuracy_valid"]) for item in robustness_items)
        if robustness_items
        else -1.0
    )
    robustness_balanced = (
        average(float(item.metrics["balanced_accuracy"]) for item in robustness_items)
        if robustness_items
        else -1.0
    )
    total_labelled = sum(int(item.metrics["total_samples"]) for item in labelled_results)
    total_valid = sum(int(item.metrics["valid_predictions"]) for item in labelled_results)
    total_correct = sum(int(item.metrics["correct_predictions"]) for item in labelled_results)
    total_errors = sum(int(item.metrics["error_count"]) for item in labelled_results)
    total_excluded = sum(int(item.metrics["excluded_predictions"]) for item in labelled_results)
    stability_total = total_labelled + (scope_result.total_samples if scope_result else 0)
    stability_errors = total_errors + (scope_result.error_count if scope_result else 0)
    stability = safe_div(stability_total - stability_errors, stability_total)

    score_parts = [
        score_metric(in_domain, 0.95, 0.90),
        score_metric(cross_scene, 0.90, 0.80),
        score_metric(stability, 0.98, 0.95),
    ]
    if robustness_items:
        score_parts.append(score_metric(robustness_accuracy, 0.85, 0.75))
    scope_rejection = None
    if scope_result is not None:
        scope_rejection = scope_result.rejection_rate
        score_parts.append(score_metric(scope_rejection, 0.60, 0.35))

    overall_score = average(score_parts)
    if overall_score >= 0.90:
        overall_grade = "Strong"
    elif overall_score >= 0.72:
        overall_grade = "Usable"
    else:
        overall_grade = "Needs Optimization"

    if robustness_items and cross_scene >= 0.90 and robustness_accuracy >= 0.85:
        conclusion = "The detector shows convincing cross-scene stability for an undergraduate project."
    elif robustness_items and cross_scene >= 0.80 and robustness_accuracy >= 0.75:
        conclusion = "The detector is usable for undergraduate project defense, but robustness still has visible room for improvement."
    elif cross_scene >= 0.80:
        conclusion = "The detector is usable for undergraduate project defense, but the robustness suites were skipped."
    else:
        conclusion = "The detector can be demonstrated, but cross-scene robustness should be described conservatively."

    return {
        "report_type": "practical_validation",
        "overall_grade": overall_grade,
        "overall_score": round(overall_score, 4),
        "overall_conclusion": conclusion,
        "total_samples": total_labelled,
        "valid_predictions": total_valid,
        "correct_predictions": total_correct,
        "error_count": total_errors,
        "excluded_predictions": total_excluded,
        "accuracy_all": safe_div(total_correct, total_labelled - total_excluded),
        "accuracy_valid": safe_div(total_correct, total_valid),
        "balanced_accuracy": average(float(item.metrics["balanced_accuracy"]) for item in labelled_results),
        "in_domain_accuracy": in_domain,
        "cross_scene_accuracy": cross_scene,
        "robustness_accuracy": robustness_accuracy,
        "robustness_balanced_accuracy": robustness_balanced,
        "service_stability": stability,
        "scope_rejection_rate": scope_rejection if scope_rejection is not None else -1.0,
    }


def build_observations(
    overall_summary: Dict[str, float | int | str],
    labelled_results: List[LabelledSuiteResult],
    scope_result: Optional[ScopeSuiteResult],
) -> List[str]:
    metrics_by_key = {item.key: item for item in labelled_results}
    test_accuracy = float(metrics_by_key["test_in_domain"].metrics["accuracy_valid"])
    validation_accuracy = float(metrics_by_key["validation_cross_scene"].metrics["accuracy_valid"])
    gap = test_accuracy - validation_accuracy
    robustness_items = [item for item in labelled_results if item.suite_kind == "robustness"]
    weakest = min(
        robustness_items,
        key=lambda item: float(item.metrics["accuracy_valid"]),
        default=None,
    )
    observations = [
        (
            f"In-domain accuracy is {format_percent(test_accuracy)}, while cross-scene Validation accuracy is "
            f"{format_percent(validation_accuracy)}."
        ),
        f"The Test-to-Validation accuracy gap is {gap * 100:.2f} percentage points.",
        (
            f"Service stability across all labelled suites is "
            f"{format_percent(float(overall_summary['service_stability']))}."
        ),
    ]
    if weakest is not None:
        observations.append(
            f"The weakest perturbation suite is {weakest.transform_title} at "
            f"{format_percent(float(weakest.metrics['accuracy_valid']))}."
        )
    elif not robustness_items:
        observations.append("Robustness perturbation suites were skipped in this run.")
    if scope_result is not None:
        observations.append(
            f"Nobody out-of-scope rejection rate is {format_percent(scope_result.rejection_rate)}, "
            f"with {scope_result.accepted_predictions} images still accepted as in-scope."
        )
    return observations


def write_predictions_csv(
    labelled_records: Dict[str, List[PredictionRecord]],
    scope_records: List[PredictionRecord],
    target: Path,
) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "suite_name",
                "path",
                "file_name",
                "truth_label",
                "predicted_label",
                "confidence",
                "is_correct",
                "latency_ms",
                "error",
            ],
        )
        writer.writeheader()
        for suite_name, records in labelled_records.items():
            for record in records:
                writer.writerow({"suite_name": suite_name, **record.__dict__})
        for record in scope_records:
            writer.writerow({"suite_name": "nobody_scope_probe", **record.__dict__})


def write_metrics_json(
    *,
    summary: Dict[str, float | int | str],
    labelled_results: List[LabelledSuiteResult],
    scope_result: Optional[ScopeSuiteResult],
    observations: List[str],
    target: Path,
) -> None:
    payload = {
        "summary": summary,
        "generated_at": datetime.now().isoformat(),
        "observations": observations,
        "labelled_suites": [item.__dict__ for item in labelled_results],
        "scope_suite": scope_result.__dict__ if scope_result is not None else None,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_score_card(title: str, value: str, hint: str) -> str:
    return f"""
    <div class="score-card">
      <div class="score-title">{html.escape(title)}</div>
      <div class="score-value">{html.escape(value)}</div>
      <div class="score-hint">{html.escape(hint)}</div>
    </div>
    """


def render_suite_rows(labelled_results: List[LabelledSuiteResult]) -> str:
    rows = []
    for item in labelled_results:
        metrics = item.metrics
        rows.append(
            f"""
            <tr>
              <td>{html.escape(item.title)}</td>
              <td>{html.escape(item.suite_kind)}</td>
              <td>{html.escape(item.transform_title)}</td>
              <td>{int(metrics['total_samples'])}</td>
              <td>{format_percent(float(metrics['accuracy_valid']))}</td>
              <td>{format_percent(float(metrics['balanced_accuracy']))}</td>
              <td>{format_percent(safe_div(int(metrics['valid_predictions']), int(metrics['total_samples'])))}</td>
              <td>{int(metrics['error_count'])}</td>
              <td>{item.average_latency_ms:.1f}</td>
            </tr>
            """
        )
    return "".join(rows)


def render_scope_panel(scope_result: Optional[ScopeSuiteResult]) -> str:
    if scope_result is None:
        return """
        <section class="panel">
          <h2>Out-of-Scope Probe</h2>
          <p>No Nobody dataset was available, so the out-of-scope probe was skipped.</p>
        </section>
        """

    labels = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in scope_result.label_breakdown.items()
    )
    return f"""
    <section class="panel">
      <h2>Out-of-Scope Probe</h2>
      <p>Dataset: <strong>{html.escape(scope_result.dataset_root)}</strong></p>
      <div class="scope-grid">
        <div class="scope-metric">
          <span>Rejection Rate</span>
          <strong>{format_percent(scope_result.rejection_rate)}</strong>
        </div>
        <div class="scope-metric">
          <span>Accepted as In-Scope</span>
          <strong>{scope_result.accepted_predictions}</strong>
        </div>
        <div class="scope-metric">
          <span>Errors</span>
          <strong>{scope_result.error_count}</strong>
        </div>
        <div class="scope-metric">
          <span>Avg Latency</span>
          <strong>{scope_result.average_latency_ms:.1f} ms</strong>
        </div>
      </div>
      <table class="metrics-table">
        <thead><tr><th>Predicted Label</th><th>Count</th></tr></thead>
        <tbody>{labels}</tbody>
      </table>
    </section>
    """


def render_html_report(
    *,
    args: argparse.Namespace,
    summary: Dict[str, float | int | str],
    labelled_results: List[LabelledSuiteResult],
    scope_result: Optional[ScopeSuiteResult],
    observations: List[str],
) -> str:
    cards_html = "".join(
        [
            render_score_card("Overall Grade", str(summary["overall_grade"]), "Undergraduate defense readiness"),
            render_score_card("In-Domain", format_percent(float(summary["in_domain_accuracy"])), "Datasets/Test"),
            render_score_card("Cross-Scene", format_percent(float(summary["cross_scene_accuracy"])), "Datasets/Validation"),
            render_score_card(
                "Robustness Avg",
                "N/A" if float(summary["robustness_accuracy"]) < 0 else format_percent(float(summary["robustness_accuracy"])),
                "JPEG / Blur / Resize",
            ),
            render_score_card("Service Stability", format_percent(float(summary["service_stability"])), "Low request failure rate"),
            render_score_card(
                "Scope Rejection",
                "N/A" if float(summary["scope_rejection_rate"]) < 0 else format_percent(float(summary["scope_rejection_rate"])),
                "Nobody dataset",
            ),
        ]
    )
    observation_items = "".join(f"<li>{html.escape(item)}</li>" for item in observations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Practical Validation Report</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: rgba(255, 252, 246, 0.95);
      --ink: #1d2a33;
      --muted: #68727c;
      --line: #e5d9c7;
      --accent: #0f766e;
      --good: #15803d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.12), transparent 26%),
        linear-gradient(180deg, #faf7f0 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 22px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 18px 44px rgba(31, 41, 55, 0.06);
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #dff5f1;
      color: #0f766e;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 36px;
      line-height: 1.08;
    }}
    p {{
      margin: 8px 0;
      color: var(--muted);
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .hero-chip {{
      padding: 12px 14px;
      border-radius: 16px;
      background: #fff7eb;
      border: 1px solid #edd8b6;
      color: #92400e;
      font-size: 14px;
    }}
    .summary-score {{
      font-size: 54px;
      font-weight: 800;
      line-height: 1;
      margin-top: 8px;
      color: var(--accent);
    }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .score-card {{
      padding: 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fffdf8, #fdf6ea);
      border: 1px solid #eadbc7;
    }}
    .score-title {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .score-value {{
      font-size: 28px;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    .score-hint {{
      font-size: 12px;
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .metrics-table th, .metrics-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      font-size: 14px;
    }}
    .metrics-table td:last-child {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .metrics-table th {{
      background: #fbf7ef;
    }}
    .observations {{
      margin: 0;
      padding-left: 18px;
      color: var(--ink);
    }}
    .observations li {{
      margin-bottom: 10px;
      line-height: 1.5;
    }}
    .scope-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0 18px;
    }}
    .scope-metric {{
      padding: 14px;
      border-radius: 16px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
    }}
    .scope-metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .scope-metric strong {{
      font-size: 22px;
    }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      .hero, .grid, .score-grid, .scope-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel">
        <div class="eyebrow">Practical Validation</div>
        <h1>Undergraduate-ready robustness report</h1>
        <p>{html.escape(str(summary['overall_conclusion']))}</p>
        <div class="hero-grid">
          <div class="hero-chip">Mode: <strong>{html.escape(args.mode)}</strong></div>
          <div class="hero-chip">Semantic threshold: <strong>{args.semantic_threshold:.2f}</strong></div>
          <div class="hero-chip">Original sample/class: <strong>{args.sample_per_class}</strong></div>
          <div class="hero-chip">Robustness sample/class: <strong>{args.robustness_sample_per_class}</strong></div>
        </div>
      </div>
      <div class="panel">
        <div class="eyebrow">Overall</div>
        <div class="summary-score">{html.escape(str(summary['overall_grade']))}</div>
        <p>Score: <strong>{float(summary['overall_score']):.2f}</strong></p>
        <p>Total labelled samples: <strong>{int(summary['total_samples'])}</strong></p>
        <p>Valid predictions: <strong>{int(summary['valid_predictions'])}</strong></p>
        <p>Labelled request errors: <strong>{int(summary['error_count'])}</strong></p>
      </div>
    </section>

    <section class="score-grid">
      {cards_html}
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Suite Metrics</h2>
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Suite</th>
              <th>Kind</th>
              <th>Transform</th>
              <th>Samples</th>
              <th>Accuracy</th>
              <th>Balanced Acc</th>
              <th>Valid Coverage</th>
              <th>Errors</th>
              <th>Avg Latency(ms)</th>
            </tr>
          </thead>
          <tbody>{render_suite_rows(labelled_results)}</tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Defense Notes</h2>
        <ul class="observations">{observation_items}</ul>
        <div class="footer-note">
          Recommendation: describe this report as cross-scene robustness and practical validation,
          not as strict academic domain generalization.
        </div>
      </div>
    </section>

    {render_scope_panel(scope_result)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args.mode, args.endpoint, args.timeout)
    runner = DetectBenchmarkRunner(
        client=client,
        semantic_threshold=args.semantic_threshold,
        workers=args.workers,
    )

    test_root = resolve_project_path(args.test_root)
    validation_root = resolve_project_path(args.validation_root)
    nobody_root = resolve_project_path(args.nobody_root)

    labelled_results: List[LabelledSuiteResult] = []
    labelled_records: Dict[str, List[PredictionRecord]] = {}

    print(f"[RELOAD] running practical validation -> {output_dir}")
    print(f"[RELOAD] in-domain dataset: {test_root}")
    print(f"[RELOAD] cross-scene dataset: {validation_root}")
    if args.mode == "internal" and args.workers > 1:
        print("[WARN] internal mode with workers>1 may be unstable for OpenCV-based detectors")

    in_domain_result, in_domain_records = run_labelled_suite(
        key="test_in_domain",
        title="In-domain Test",
        suite_kind="in_domain",
        dataset_root=test_root,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
        runner=runner,
    )
    labelled_results.append(in_domain_result)
    labelled_records[in_domain_result.key] = in_domain_records

    validation_result, validation_records = run_labelled_suite(
        key="validation_cross_scene",
        title="Cross-scene Validation",
        suite_kind="cross_scene",
        dataset_root=validation_root,
        sample_per_class=args.sample_per_class,
        seed=args.seed + 1,
        runner=runner,
    )
    labelled_results.append(validation_result)
    labelled_records[validation_result.key] = validation_records

    if not args.skip_robustness:
        with tempfile.TemporaryDirectory(prefix="practical_validation_") as temp_dir:
            workspace = Path(temp_dir)
            for index, perturbation in enumerate(build_perturbations(), start=1):
                print(f"[RELOAD] robustness suite: {perturbation.name}")
                suite_result, suite_records = run_labelled_suite(
                    key=f"robustness_{perturbation.name}",
                    title=f"Robustness - {perturbation.title}",
                    suite_kind="robustness",
                    dataset_root=validation_root,
                    sample_per_class=args.robustness_sample_per_class,
                    seed=args.seed + 100 + index,
                    runner=runner,
                    perturbation=perturbation,
                    workspace=workspace,
                )
                labelled_results.append(suite_result)
                labelled_records[suite_result.key] = suite_records

    scope_result = None
    scope_records: List[PredictionRecord] = []
    if not args.skip_scope:
        print(f"[RELOAD] scope probe dataset: {nobody_root}")
        scope_result, scope_records = run_scope_suite(
            key="nobody_scope_probe",
            title="Nobody Scope Probe",
            dataset_root=nobody_root,
            limit=args.nobody_limit,
            seed=args.seed + 500,
            runner=runner,
        )

    overall_summary = build_overall_summary(labelled_results, scope_result)
    observations = build_observations(overall_summary, labelled_results, scope_result)

    html_path = output_dir / "index.html"
    csv_path = output_dir / "predictions.csv"
    json_path = output_dir / "metrics.json"

    html_path.write_text(
        render_html_report(
            args=args,
            summary=overall_summary,
            labelled_results=labelled_results,
            scope_result=scope_result,
            observations=observations,
        ),
        encoding="utf-8",
    )
    write_predictions_csv(labelled_records, scope_records, csv_path)
    write_metrics_json(
        summary=overall_summary,
        labelled_results=labelled_results,
        scope_result=scope_result,
        observations=observations,
        target=json_path,
    )

    print(f"[CREATE] report written: {html_path}")
    print(f"[CREATE] csv written: {csv_path}")
    print(f"[CREATE] json written: {json_path}")
    print(
        "[RELOAD] summary: "
        f"grade={overall_summary['overall_grade']}, "
        f"in_domain={format_percent(float(overall_summary['in_domain_accuracy']))}, "
        f"cross_scene={format_percent(float(overall_summary['cross_scene_accuracy']))}, "
        f"robustness={'N/A' if float(overall_summary['robustness_accuracy']) < 0 else format_percent(float(overall_summary['robustness_accuracy']))}"
    )


if __name__ == "__main__":
    main()
