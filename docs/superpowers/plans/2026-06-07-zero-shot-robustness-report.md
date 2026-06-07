# Zero-shot Robustness Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible zero-shot and black-box perturbation robustness benchmark that writes JSON, CSV, HTML, and Chinese Markdown reports without changing model weights or `/detect`.

**Architecture:** Create one focused benchmark script under `scripts/benchmark/` that reuses `visualize_detect_benchmark.py` clients, sampling, prediction records, summary metrics, and audit summaries. Keep image perturbation, metric deltas, recommendations, report rendering, and CLI orchestration in this script so the existing detection path remains unchanged. Add tests for pure logic and a fake-client smoke path so most behavior is verified without running heavy models.

**Tech Stack:** Python, Pillow, pytest, existing benchmark helpers, existing Flask detect facade through `InternalDetectClient` for real runs.

---

## File Structure

- Create: `scripts/benchmark/visualize_zero_shot_robustness.py`
  - Owns CLI parsing, perturbation transforms, clean/perturbed suite execution, metric drop calculation, recommendations, JSON/CSV/HTML/Markdown output.
  - Imports and reuses `DatasetSample`, `DetectBenchmarkRunner`, `InternalDetectClient`, `HttpDetectClient`, `PredictionRecord`, `collect_samples`, `compute_summary`, `compute_audit_summary`, `format_percent`, `safe_div`, and `VALID_LABELS`.
- Create: `tests/test_zero_shot_robustness_report.py`
  - Tests perturbation helpers, metric drop calculation, recommendation rules, payload shape, Markdown generation, and a fake-client end-to-end report write.
- No changes: `app.py`, `service/`, `detector_config.py`, model weights, mapping files.

---

### Task 1: Add Pure Metric and Recommendation Helpers

**Files:**
- Create: `scripts/benchmark/visualize_zero_shot_robustness.py`
- Create: `tests/test_zero_shot_robustness_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_zero_shot_robustness_report.py` with:

```python
from scripts.benchmark.visualize_zero_shot_robustness import (
    build_recommendations,
    compute_metric_drops,
    normalize_domain_name,
    parse_key_value_items,
)


def test_parse_key_value_items_and_domain_normalization():
    parsed = parse_key_value_items(
        ["DFDC=Datasets/DFDC", "WildDeepfake=Datasets/WildDeepfake"]
    )

    assert parsed == {
        "DFDC": "Datasets/DFDC",
        "WildDeepfake": "Datasets/WildDeepfake",
    }
    assert normalize_domain_name("Celeb-DF") == "celeb_df"
    assert normalize_domain_name("WildDeepfake") == "wilddeepfake"


def test_compute_metric_drops_rounds_clean_minus_perturbed():
    clean_metrics = {
        "accuracy_valid": 0.92,
        "balanced_accuracy": 0.90,
        "recall_fake": 0.88,
        "specificity_real": 0.94,
    }
    perturbed_metrics = {
        "accuracy_valid": 0.81,
        "balanced_accuracy": 0.80,
        "recall_fake": 0.73,
        "specificity_real": 0.87,
    }
    clean_audit = {
        "evidence_hit_rate": 0.42,
        "high_score_no_evidence_rate": 0.10,
        "joint_evidence_correct_rate": 0.36,
    }
    perturbed_audit = {
        "evidence_hit_rate": 0.31,
        "high_score_no_evidence_rate": 0.22,
        "joint_evidence_correct_rate": 0.25,
    }

    drops = compute_metric_drops(
        clean_metrics=clean_metrics,
        perturbed_metrics=perturbed_metrics,
        clean_audit=clean_audit,
        perturbed_audit=perturbed_audit,
    )

    assert drops["accuracy_drop"] == 0.11
    assert drops["balanced_accuracy_drop"] == 0.10
    assert drops["fake_recall_drop"] == 0.15
    assert drops["specificity_drop"] == 0.07
    assert drops["evidence_hit_rate_drop"] == 0.11
    assert drops["high_score_no_evidence_rate_delta"] == 0.12
    assert drops["joint_evidence_correct_rate_drop"] == 0.11


def test_build_recommendations_flags_domain_and_evidence_weaknesses():
    domain_suites = [
        {
            "domain": "DFDC",
            "metrics": {
                "balanced_accuracy": 0.82,
                "recall_fake": 0.76,
            },
        }
    ]
    perturbation_suites = [
        {
            "domain": "DFDC",
            "perturbation": "jpeg_q60",
            "drops": {
                "accuracy_drop": 0.10,
                "evidence_hit_rate_drop": 0.12,
                "high_score_no_evidence_rate_delta": 0.09,
            },
        }
    ]

    recommendations = build_recommendations(domain_suites, perturbation_suites)

    assert recommendations == [
        {
            "priority": "P1",
            "target": "DFDC",
            "reason": "clean balanced accuracy is below 90.00%",
            "suggestion": "Collect DFDC hard cases before retraining.",
        },
        {
            "priority": "P1",
            "target": "DFDC",
            "reason": "fake recall is below 85.00%",
            "suggestion": "Add more fake hard cases for DFDC to reduce missed forgeries.",
        },
        {
            "priority": "P1",
            "target": "DFDC/jpeg_q60",
            "reason": "accuracy drop is at least 8.00%",
            "suggestion": "Add jpeg_q60 augmentation samples before retraining.",
        },
        {
            "priority": "P2",
            "target": "DFDC/jpeg_q60",
            "reason": "high-score samples without evidence increased by at least 8.00%",
            "suggestion": "Review detector feature to mapping coverage for this perturbation.",
        },
        {
            "priority": "P2",
            "target": "DFDC/jpeg_q60",
            "reason": "evidence hit rate dropped by at least 10.00%",
            "suggestion": "Inspect feature activation stability after this perturbation.",
        },
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.benchmark.visualize_zero_shot_robustness'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/benchmark/visualize_zero_shot_robustness.py` with:

```python
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.visualize_detect_benchmark import (
    DatasetSample,
    DetectBenchmarkRunner,
    HttpDetectClient,
    InternalDetectClient,
    PredictionRecord,
    VALID_LABELS,
    collect_samples,
    compute_audit_summary,
    compute_summary,
    format_percent,
    safe_div,
)


RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    title: str
    description: str
    transform: Callable[[Image.Image, random.Random], Image.Image]
    jpeg_quality: int = 92


@dataclass(frozen=True)
class SuiteResult:
    domain: str
    dataset_root: str
    suite_key: str
    suite_kind: str
    decision_profile: str
    perturbation: str
    sample_count: int
    metrics: Dict[str, float | int]
    audit_summary: Dict[str, object]
    average_latency_ms: float


def parse_key_value_items(items: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE item, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Expected non-empty KEY=VALUE item, got: {item}")
        parsed[key] = value
    return parsed


def normalize_domain_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def average(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def summary_to_metrics(summary) -> Dict[str, float | int]:
    return {
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
        "fake_support": summary.fake_support,
        "real_support": summary.real_support,
        "tp": summary.tp,
        "tn": summary.tn,
        "fp": summary.fp,
        "fn": summary.fn,
    }


def rounded_delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)


def compute_metric_drops(
    *,
    clean_metrics: Dict[str, float | int],
    perturbed_metrics: Dict[str, float | int],
    clean_audit: Dict[str, object],
    perturbed_audit: Dict[str, object],
) -> Dict[str, float]:
    return {
        "accuracy_drop": rounded_delta(
            float(clean_metrics.get("accuracy_valid", 0.0)),
            float(perturbed_metrics.get("accuracy_valid", 0.0)),
        ),
        "balanced_accuracy_drop": rounded_delta(
            float(clean_metrics.get("balanced_accuracy", 0.0)),
            float(perturbed_metrics.get("balanced_accuracy", 0.0)),
        ),
        "fake_recall_drop": rounded_delta(
            float(clean_metrics.get("recall_fake", 0.0)),
            float(perturbed_metrics.get("recall_fake", 0.0)),
        ),
        "specificity_drop": rounded_delta(
            float(clean_metrics.get("specificity_real", 0.0)),
            float(perturbed_metrics.get("specificity_real", 0.0)),
        ),
        "evidence_hit_rate_drop": rounded_delta(
            float(clean_audit.get("evidence_hit_rate", 0.0)),
            float(perturbed_audit.get("evidence_hit_rate", 0.0)),
        ),
        "high_score_no_evidence_rate_delta": rounded_delta(
            float(perturbed_audit.get("high_score_no_evidence_rate", 0.0)),
            float(clean_audit.get("high_score_no_evidence_rate", 0.0)),
        ),
        "joint_evidence_correct_rate_drop": rounded_delta(
            float(clean_audit.get("joint_evidence_correct_rate", 0.0)),
            float(perturbed_audit.get("joint_evidence_correct_rate", 0.0)),
        ),
    }


def build_recommendations(
    domain_suites: Sequence[Dict],
    perturbation_suites: Sequence[Dict],
) -> List[Dict[str, str]]:
    recommendations: List[Dict[str, str]] = []
    for suite in domain_suites:
        domain = str(suite["domain"])
        metrics = suite.get("metrics", {})
        if float(metrics.get("balanced_accuracy", 0.0)) < 0.90:
            recommendations.append(
                {
                    "priority": "P1",
                    "target": domain,
                    "reason": "clean balanced accuracy is below 90.00%",
                    "suggestion": f"Collect {domain} hard cases before retraining.",
                }
            )
        if float(metrics.get("recall_fake", 0.0)) < 0.85:
            recommendations.append(
                {
                    "priority": "P1",
                    "target": domain,
                    "reason": "fake recall is below 85.00%",
                    "suggestion": f"Add more fake hard cases for {domain} to reduce missed forgeries.",
                }
            )

    for suite in perturbation_suites:
        domain = str(suite["domain"])
        perturbation = str(suite["perturbation"])
        target = f"{domain}/{perturbation}"
        drops = suite.get("drops", {})
        if float(drops.get("accuracy_drop", 0.0)) >= 0.08:
            recommendations.append(
                {
                    "priority": "P1",
                    "target": target,
                    "reason": "accuracy drop is at least 8.00%",
                    "suggestion": f"Add {perturbation} augmentation samples before retraining.",
                }
            )
        if float(drops.get("high_score_no_evidence_rate_delta", 0.0)) >= 0.08:
            recommendations.append(
                {
                    "priority": "P2",
                    "target": target,
                    "reason": "high-score samples without evidence increased by at least 8.00%",
                    "suggestion": "Review detector feature to mapping coverage for this perturbation.",
                }
            )
        if float(drops.get("evidence_hit_rate_drop", 0.0)) >= 0.10:
            recommendations.append(
                {
                    "priority": "P2",
                    "target": target,
                    "reason": "evidence hit rate dropped by at least 10.00%",
                    "suggestion": "Inspect feature activation stability after this perturbation.",
                }
            )
    return recommendations
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/visualize_zero_shot_robustness.py tests/test_zero_shot_robustness_report.py
git commit -m "feat: add zero-shot robustness metric helpers"
```

---

### Task 2: Add Perturbation Generation and Suite Execution

**Files:**
- Modify: `scripts/benchmark/visualize_zero_shot_robustness.py`
- Modify: `tests/test_zero_shot_robustness_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zero_shot_robustness_report.py`:

```python
from pathlib import Path

from PIL import Image

from scripts.benchmark.visualize_detect_benchmark import DatasetSample, PredictionRecord
from scripts.benchmark.visualize_zero_shot_robustness import (
    build_perturbations,
    materialize_perturbed_samples,
    run_suite,
)


class FakeDetectClient:
    def predict(
        self,
        image_path: Path,
        semantic_threshold: float,
        decision_profile: str | None,
        decision_threshold_override: float | None,
    ) -> dict:
        label = "FAKE" if "Fake" in image_path.parts else "REAL"
        return {
            "label": label,
            "confidence": 0.91,
            "decision_fake_score": 0.82 if label == "FAKE" else 0.12,
            "decision_threshold": 0.5,
            "score_source": "fake_client",
            "threshold_source": "test",
            "decision_profile": decision_profile or "",
            "reasoning_type": "test_reasoning",
            "risk_level": "low",
            "needs_review": False,
            "review_reasons": [],
            "diagnostic_chain": ["input", "detect", "output"],
            "evidence": [{"name": "synthetic evidence"}] if label == "FAKE" else [],
            "evidence_diagnostics": {
                "requested_subdomains": 1 if label == "FAKE" else 0,
                "unresolved_subdomains": 0,
            },
            "evidence_alignment_score": 0.7 if label == "FAKE" else 0.0,
            "graph_influence_weight": 0.2,
        }


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_materialize_perturbed_samples_preserves_labels_and_size(tmp_path):
    fake_path = tmp_path / "source" / "Fake" / "fake.jpg"
    real_path = tmp_path / "source" / "Real" / "real.jpg"
    _write_image(fake_path, (240, 20, 20))
    _write_image(real_path, (20, 240, 20))
    samples = [
        DatasetSample(path=fake_path, truth_label="FAKE"),
        DatasetSample(path=real_path, truth_label="REAL"),
    ]
    perturbation = build_perturbations(["crop_restore"])[0]

    generated = materialize_perturbed_samples(
        samples=samples,
        perturbation=perturbation,
        workspace=tmp_path / "workspace",
        seed=7,
    )

    assert [item.truth_label for item in generated] == ["FAKE", "REAL"]
    assert generated[0].path.exists()
    with Image.open(generated[0].path) as image:
        assert image.size == (32, 32)


def test_run_suite_uses_existing_runner_and_returns_audit_metrics(tmp_path):
    fake_path = tmp_path / "Fake" / "fake.jpg"
    real_path = tmp_path / "Real" / "real.jpg"
    _write_image(fake_path, (240, 20, 20))
    _write_image(real_path, (20, 240, 20))
    samples = [
        DatasetSample(path=fake_path, truth_label="FAKE"),
        DatasetSample(path=real_path, truth_label="REAL"),
    ]

    suite, records = run_suite(
        domain="Unit",
        dataset_root=tmp_path,
        suite_key="clean__Unit",
        suite_kind="clean",
        decision_profile="unit_profile",
        samples=samples,
        client=FakeDetectClient(),
        semantic_threshold=0.8,
        workers=1,
    )

    assert suite.domain == "Unit"
    assert suite.metrics["accuracy_valid"] == 1.0
    assert suite.audit_summary["reasoning_type_coverage"] == 1.0
    assert suite.audit_summary["diagnostic_chain_coverage"] == 1.0
    assert suite.audit_summary["fake_evidence_hit_rate"] == 1.0
    assert [record.predicted_label for record in records] == ["FAKE", "REAL"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
```

Expected: FAIL with import errors for `build_perturbations`, `materialize_perturbed_samples`, and `run_suite`.

- [ ] **Step 3: Add perturbation and suite execution code**

Append these functions to `scripts/benchmark/visualize_zero_shot_robustness.py` after `build_recommendations`:

```python
def jpeg_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    return image.copy()


def blur_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=1.2))


def resize_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    down_width = max(64, int(width * 0.60))
    down_height = max(64, int(height * 0.60))
    reduced = image.resize((down_width, down_height), RESAMPLING.BILINEAR)
    return reduced.resize((width, height), RESAMPLING.BICUBIC)


def crop_restore_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    crop_x = max(1, int(width * 0.08))
    crop_y = max(1, int(height * 0.08))
    cropped = image.crop((crop_x, crop_y, width - crop_x, height - crop_y))
    return cropped.resize((width, height), RESAMPLING.BICUBIC)


def color_shift_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    bright = ImageEnhance.Brightness(image).enhance(1.18)
    return ImageEnhance.Color(bright).enhance(0.82)


def light_noise_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    pixels = image.convert("RGB").load()
    output = image.convert("RGB").copy()
    output_pixels = output.load()
    width, height = output.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            delta = rng.randint(-8, 8)
            output_pixels[x, y] = (
                max(0, min(255, r + delta)),
                max(0, min(255, g + delta)),
                max(0, min(255, b + delta)),
            )
    return output


def occlusion_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    output = image.convert("RGB").copy()
    width, height = output.size
    box_w = max(8, int(width * 0.22))
    box_h = max(8, int(height * 0.16))
    left = max(0, int(width * 0.58) - box_w // 2)
    top = max(0, int(height * 0.58) - box_h // 2)
    right = min(width, left + box_w)
    bottom = min(height, top + box_h)
    pixels = output.load()
    for y in range(top, bottom):
        for x in range(left, right):
            pixels[x, y] = (28, 28, 28)
    return output


def all_perturbations() -> Dict[str, PerturbationSpec]:
    return {
        "jpeg_q60": PerturbationSpec(
            name="jpeg_q60",
            title="JPEG Q60",
            description="Simulate social-platform recompression at JPEG quality 60.",
            transform=jpeg_transform,
            jpeg_quality=60,
        ),
        "gaussian_blur": PerturbationSpec(
            name="gaussian_blur",
            title="Gaussian Blur",
            description="Apply mild blur to simulate low-quality capture or reposting.",
            transform=blur_transform,
        ),
        "downscale_restore": PerturbationSpec(
            name="downscale_restore",
            title="Downscale + Restore",
            description="Shrink to 60% and restore to the original resolution.",
            transform=resize_transform,
        ),
        "crop_restore": PerturbationSpec(
            name="crop_restore",
            title="Crop + Restore",
            description="Crop 8% border and restore size to simulate partial framing loss.",
            transform=crop_restore_transform,
        ),
        "color_shift": PerturbationSpec(
            name="color_shift",
            title="Brightness / Color Shift",
            description="Apply brightness and color changes to simulate filters.",
            transform=color_shift_transform,
        ),
        "light_noise": PerturbationSpec(
            name="light_noise",
            title="Light Noise",
            description="Apply small random pixel noise as a black-box stress test.",
            transform=light_noise_transform,
        ),
        "occlusion": PerturbationSpec(
            name="occlusion",
            title="Local Occlusion",
            description="Add a small dark rectangle to simulate sticker, watermark, or occlusion.",
            transform=occlusion_transform,
        ),
    }


def build_perturbations(names: Optional[Sequence[str]] = None) -> List[PerturbationSpec]:
    available = all_perturbations()
    selected_names = list(names) if names else list(available)
    unknown = [name for name in selected_names if name not in available]
    if unknown:
        raise ValueError(f"Unknown perturbations: {', '.join(unknown)}")
    return [available[name] for name in selected_names]


def materialize_perturbed_samples(
    *,
    samples: Sequence[DatasetSample],
    perturbation: PerturbationSpec,
    workspace: Path,
    seed: int,
) -> List[DatasetSample]:
    rng = random.Random(seed)
    generated: List[DatasetSample] = []
    target_root = workspace / perturbation.name
    target_root.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples, start=1):
        class_dir = target_root / sample.truth_label
        class_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_dir / f"{sample.path.stem}_{perturbation.name}_{index:04d}.jpg"
        with Image.open(sample.path) as image:
            transformed = perturbation.transform(image.convert("RGB"), rng)
            transformed.save(
                output_path,
                format="JPEG",
                quality=perturbation.jpeg_quality,
                optimize=False,
            )
        generated.append(DatasetSample(path=output_path, truth_label=sample.truth_label))
    return generated


def run_suite(
    *,
    domain: str,
    dataset_root: Path,
    suite_key: str,
    suite_kind: str,
    decision_profile: str,
    samples: Sequence[DatasetSample],
    client,
    semantic_threshold: float,
    workers: int,
    perturbation: str = "",
) -> tuple[SuiteResult, List[PredictionRecord]]:
    runner = DetectBenchmarkRunner(
        client=client,
        semantic_threshold=semantic_threshold,
        decision_profile=decision_profile or None,
        workers=workers,
    )
    records = runner.run(samples)
    summary = compute_summary(records)
    metrics = summary_to_metrics(summary)
    audit_summary = compute_audit_summary(records)
    suite = SuiteResult(
        domain=domain,
        dataset_root=str(dataset_root),
        suite_key=suite_key,
        suite_kind=suite_kind,
        decision_profile=decision_profile,
        perturbation=perturbation,
        sample_count=len(records),
        metrics=metrics,
        audit_summary=audit_summary,
        average_latency_ms=average(record.latency_ms for record in records),
    )
    return suite, records
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/visualize_zero_shot_robustness.py tests/test_zero_shot_robustness_report.py
git commit -m "feat: run zero-shot robustness suites"
```

---

### Task 3: Add Payload, CSV, Markdown, and HTML Report Writers

**Files:**
- Modify: `scripts/benchmark/visualize_zero_shot_robustness.py`
- Modify: `tests/test_zero_shot_robustness_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zero_shot_robustness_report.py`:

```python
from scripts.benchmark.visualize_zero_shot_robustness import (
    build_report_payload,
    render_html_report,
    render_markdown_summary,
    write_predictions_csv,
)


def _suite_dict(domain: str, accuracy: float, perturbation: str = "") -> dict:
    return {
        "domain": domain,
        "dataset_root": f"Datasets/{domain}",
        "suite_key": f"clean__{domain}" if not perturbation else f"perturbed__{domain}__{perturbation}",
        "suite_kind": "clean" if not perturbation else "perturbed",
        "decision_profile": normalize_domain_name(domain),
        "perturbation": perturbation,
        "sample_count": 2,
        "metrics": {
            "total_samples": 2,
            "valid_predictions": 2,
            "correct_predictions": int(accuracy * 2),
            "accuracy_valid": accuracy,
            "balanced_accuracy": accuracy,
            "recall_fake": accuracy,
            "specificity_real": accuracy,
        },
        "audit_summary": {
            "reasoning_type_coverage": 1.0,
            "diagnostic_chain_coverage": 1.0,
            "evidence_hit_rate": 0.5,
            "high_score_no_evidence_rate": 0.1,
            "joint_evidence_correct_rate": 0.5,
        },
        "average_latency_ms": 5.0,
    }


def test_build_report_payload_contains_summary_and_recommendations():
    clean = [_suite_dict("DFDC", 0.82)]
    perturbed = [
        {
            **_suite_dict("DFDC", 0.72, "jpeg_q60"),
            "clean_reference": clean[0]["metrics"],
            "drops": {"accuracy_drop": 0.10},
        }
    ]

    payload = build_report_payload(
        config={"sample_per_class": 1},
        domain_suites=clean,
        perturbation_suites=perturbed,
        warnings=["sample warning"],
    )

    assert payload["report_type"] == "zero_shot_robustness"
    assert payload["summary"]["domain_count"] == 1
    assert payload["summary"]["worst_domain"] == "DFDC"
    assert payload["summary"]["worst_perturbation"] == "DFDC/jpeg_q60"
    assert payload["warnings"] == ["sample warning"]
    assert payload["recommendations"][0]["target"] == "DFDC"


def test_render_markdown_summary_uses_conservative_language():
    payload = build_report_payload(
        config={"sample_per_class": 1},
        domain_suites=[_suite_dict("DFDC", 0.82)],
        perturbation_suites=[],
        warnings=[],
    )

    markdown = render_markdown_summary(payload)

    assert "# Zero-shot 与扰动鲁棒性评测摘要" in markdown
    assert "工程抽样口径" in markdown
    assert "不等价于论文官方 protocol" in markdown
    assert "DFDC" in markdown


def test_write_predictions_csv_includes_suite_name(tmp_path):
    record = PredictionRecord(
        path="/tmp/fake.jpg",
        file_name="fake.jpg",
        truth_label="FAKE",
        predicted_label="FAKE",
        confidence=0.9,
        is_correct=True,
        latency_ms=1.5,
    )
    output_path = tmp_path / "predictions.csv"

    write_predictions_csv({"clean__Unit": [record]}, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "suite_name,path,file_name,truth_label" in text
    assert "clean__Unit,/tmp/fake.jpg,fake.jpg,FAKE" in text


def test_render_html_report_contains_domain_and_perturbation_tables():
    payload = build_report_payload(
        config={"sample_per_class": 1},
        domain_suites=[_suite_dict("DFDC", 0.82)],
        perturbation_suites=[
            {
                **_suite_dict("DFDC", 0.72, "jpeg_q60"),
                "clean_reference": {"accuracy_valid": 0.82},
                "drops": {"accuracy_drop": 0.10},
            }
        ],
        warnings=[],
    )

    html_text = render_html_report(payload)

    assert "<title>Zero-shot Robustness Report</title>" in html_text
    assert "Zero-shot 域级表现" in html_text
    assert "扰动鲁棒性" in html_text
    assert "DFDC" in html_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
```

Expected: FAIL with import errors for report writer functions.

- [ ] **Step 3: Add report writer implementation**

Append to `scripts/benchmark/visualize_zero_shot_robustness.py`:

```python
def suite_to_dict(suite: SuiteResult) -> Dict:
    return asdict(suite)


def build_overall_summary(
    domain_suites: Sequence[Dict],
    perturbation_suites: Sequence[Dict],
) -> Dict[str, object]:
    clean_acc = [float(item["metrics"].get("accuracy_valid", 0.0)) for item in domain_suites]
    perturbed_acc = [
        float(item["metrics"].get("accuracy_valid", 0.0)) for item in perturbation_suites
    ]
    worst_domain_item = min(
        domain_suites,
        key=lambda item: float(item["metrics"].get("balanced_accuracy", 0.0)),
        default=None,
    )
    worst_perturbation_item = max(
        perturbation_suites,
        key=lambda item: float(item.get("drops", {}).get("accuracy_drop", 0.0)),
        default=None,
    )
    return {
        "domain_count": len(domain_suites),
        "perturbation_suite_count": len(perturbation_suites),
        "mean_clean_accuracy": average(clean_acc),
        "mean_perturbed_accuracy": average(perturbed_acc),
        "max_accuracy_drop": (
            float(worst_perturbation_item.get("drops", {}).get("accuracy_drop", 0.0))
            if worst_perturbation_item
            else 0.0
        ),
        "worst_domain": str(worst_domain_item["domain"]) if worst_domain_item else "",
        "worst_perturbation": (
            f"{worst_perturbation_item['domain']}/{worst_perturbation_item['perturbation']}"
            if worst_perturbation_item
            else ""
        ),
    }


def build_report_payload(
    *,
    config: Dict,
    domain_suites: Sequence[Dict],
    perturbation_suites: Sequence[Dict],
    warnings: Sequence[str],
) -> Dict:
    return {
        "report_type": "zero_shot_robustness",
        "generated_at": datetime.now().isoformat(),
        "config": dict(config),
        "summary": build_overall_summary(domain_suites, perturbation_suites),
        "domain_suites": list(domain_suites),
        "perturbation_suites": list(perturbation_suites),
        "recommendations": build_recommendations(domain_suites, perturbation_suites),
        "warnings": list(warnings),
    }


def render_markdown_summary(payload: Dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Zero-shot 与扰动鲁棒性评测摘要",
        "",
        "本报告采用工程抽样口径，用于评估当前检测链路面对未知数据域和黑盒近似扰动时的稳定性；不等价于论文官方 protocol。",
        "",
        "## 总览",
        "",
        f"- 覆盖数据域：{summary['domain_count']}",
        f"- 平均 clean accuracy：{format_percent(float(summary['mean_clean_accuracy']))}",
        f"- 平均扰动 accuracy：{format_percent(float(summary['mean_perturbed_accuracy']))}",
        f"- 最大 accuracy drop：{format_percent(float(summary['max_accuracy_drop']))}",
        f"- 最弱 clean 数据域：{summary['worst_domain'] or 'N/A'}",
        f"- 最脆弱扰动：{summary['worst_perturbation'] or 'N/A'}",
        "",
        "## Zero-shot 域级表现",
        "",
        "| 数据域 | Accuracy | Balanced Acc | Fake Recall | Specificity | Evidence Hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for suite in payload["domain_suites"]:
        metrics = suite["metrics"]
        audit = suite["audit_summary"]
        lines.append(
            "| {domain} | {acc} | {bal} | {recall} | {spec} | {evidence} |".format(
                domain=suite["domain"],
                acc=format_percent(float(metrics.get("accuracy_valid", 0.0))),
                bal=format_percent(float(metrics.get("balanced_accuracy", 0.0))),
                recall=format_percent(float(metrics.get("recall_fake", 0.0))),
                spec=format_percent(float(metrics.get("specificity_real", 0.0))),
                evidence=format_percent(float(audit.get("evidence_hit_rate", 0.0))),
            )
        )
    lines.extend(["", "## 训练建议", ""])
    recommendations = payload.get("recommendations", [])
    if recommendations:
        for item in recommendations:
            lines.append(
                f"- [{item['priority']}] {item['target']}：{item['suggestion']} 原因：{item['reason']}。"
            )
    else:
        lines.append("- 当前抽样口径下未触发自动训练建议。")
    warnings = payload.get("warnings", [])
    if warnings:
        lines.extend(["", "## 警告", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def render_html_report(payload: Dict) -> str:
    summary = payload["summary"]
    domain_rows = []
    for suite in payload["domain_suites"]:
        metrics = suite["metrics"]
        audit = suite["audit_summary"]
        domain_rows.append(
            f"<tr><td>{html.escape(str(suite['domain']))}</td>"
            f"<td>{html.escape(str(suite.get('decision_profile') or 'no-profile'))}</td>"
            f"<td>{format_percent(float(metrics.get('accuracy_valid', 0.0)))}</td>"
            f"<td>{format_percent(float(metrics.get('balanced_accuracy', 0.0)))}</td>"
            f"<td>{format_percent(float(metrics.get('recall_fake', 0.0)))}</td>"
            f"<td>{format_percent(float(audit.get('evidence_hit_rate', 0.0)))}</td></tr>"
        )
    perturbation_rows = []
    for suite in payload["perturbation_suites"]:
        metrics = suite["metrics"]
        drops = suite.get("drops", {})
        perturbation_rows.append(
            f"<tr><td>{html.escape(str(suite['domain']))}</td>"
            f"<td>{html.escape(str(suite['perturbation']))}</td>"
            f"<td>{format_percent(float(metrics.get('accuracy_valid', 0.0)))}</td>"
            f"<td>{format_percent(float(drops.get('accuracy_drop', 0.0)))}</td>"
            f"<td>{format_percent(float(drops.get('fake_recall_drop', 0.0)))}</td>"
            f"<td>{format_percent(float(drops.get('high_score_no_evidence_rate_delta', 0.0)))}</td></tr>"
        )
    recommendation_items = "".join(
        f"<li><strong>{html.escape(item['target'])}</strong>: {html.escape(item['suggestion'])}</li>"
        for item in payload.get("recommendations", [])
    ) or "<li>当前抽样口径下未触发自动训练建议。</li>"
    warning_items = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in payload.get("warnings", [])
    ) or "<li>无。</li>"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Zero-shot Robustness Report</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", "PingFang SC", sans-serif; background: #f7f8fb; color: #172033; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 44px; }}
    .hero, .panel {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 14px; }}
    .card span {{ display: block; color: #667085; font-size: 12px; margin-bottom: 6px; }}
    .card strong {{ font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e5eaf2; padding: 9px 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f0f4f9; }}
    td:not(:first-child), th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin-bottom: 8px; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Zero-shot 与扰动鲁棒性评测报告</h1>
      <p>工程抽样口径；用于评估未知域和黑盒近似扰动下的检测稳定性，不等价于论文官方 protocol。</p>
    </section>
    <section class="cards">
      <div class="card"><span>平均 Clean Accuracy</span><strong>{format_percent(float(summary['mean_clean_accuracy']))}</strong></div>
      <div class="card"><span>平均扰动 Accuracy</span><strong>{format_percent(float(summary['mean_perturbed_accuracy']))}</strong></div>
      <div class="card"><span>最大 Accuracy Drop</span><strong>{format_percent(float(summary['max_accuracy_drop']))}</strong></div>
      <div class="card"><span>最弱域</span><strong>{html.escape(str(summary['worst_domain'] or 'N/A'))}</strong></div>
    </section>
    <section class="panel">
      <h2>Zero-shot 域级表现</h2>
      <table>
        <thead><tr><th>数据域</th><th>Profile</th><th>Accuracy</th><th>Balanced Acc</th><th>Fake Recall</th><th>Evidence Hit</th></tr></thead>
        <tbody>{''.join(domain_rows)}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2>扰动鲁棒性</h2>
      <table>
        <thead><tr><th>数据域</th><th>扰动</th><th>Accuracy</th><th>Accuracy Drop</th><th>Fake Recall Drop</th><th>高分无证据增量</th></tr></thead>
        <tbody>{''.join(perturbation_rows)}</tbody>
      </table>
    </section>
    <section class="panel"><h2>训练建议</h2><ul>{recommendation_items}</ul></section>
    <section class="panel"><h2>警告</h2><ul>{warning_items}</ul></section>
  </div>
</body>
</html>
"""


def write_predictions_csv(records_by_suite: Dict[str, Sequence[PredictionRecord]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite_name",
        "path",
        "file_name",
        "truth_label",
        "predicted_label",
        "confidence",
        "is_correct",
        "latency_ms",
        "decision_fake_score",
        "decision_threshold",
        "decision_margin",
        "score_source",
        "threshold_source",
        "decision_profile",
        "reasoning_type",
        "risk_level",
        "needs_review",
        "review_reasons_count",
        "diagnostic_chain_len",
        "evidence_count",
        "evidence_requested",
        "evidence_unresolved",
        "evidence_alignment_score",
        "graph_influence_weight",
        "error",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for suite_name, records in records_by_suite.items():
            for record in records:
                writer.writerow({"suite_name": suite_name, **asdict(record)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
```

Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/visualize_zero_shot_robustness.py tests/test_zero_shot_robustness_report.py
git commit -m "feat: render zero-shot robustness reports"
```

---

### Task 4: Add CLI Orchestration and End-to-End Fake-Client Test

**Files:**
- Modify: `scripts/benchmark/visualize_zero_shot_robustness.py`
- Modify: `tests/test_zero_shot_robustness_report.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zero_shot_robustness_report.py`:

```python
from scripts.benchmark.visualize_zero_shot_robustness import run_report


def test_run_report_with_fake_client_writes_all_outputs(tmp_path):
    dataset_root = tmp_path / "Datasets" / "Unit"
    _write_image(dataset_root / "Fake" / "fake1.jpg", (240, 20, 20))
    _write_image(dataset_root / "Fake" / "fake2.jpg", (230, 30, 30))
    _write_image(dataset_root / "Real" / "real1.jpg", (20, 240, 20))
    _write_image(dataset_root / "Real" / "real2.jpg", (30, 230, 30))
    output_dir = tmp_path / "reports" / "zero_shot"

    payload = run_report(
        datasets={"Unit": str(dataset_root)},
        decision_profiles={"Unit": "unit_profile"},
        sample_per_class=1,
        robustness_sample_per_class=1,
        perturbation_names=["jpeg_q60"],
        output_dir=output_dir,
        client=FakeDetectClient(),
        semantic_threshold=0.8,
        seed=11,
        workers=1,
        skip_robustness=False,
    )

    assert payload["summary"]["domain_count"] == 1
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "index.html").exists()
    assert (output_dir / "summary.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py::test_run_report_with_fake_client_writes_all_outputs -q
```

Expected: FAIL with import error for `run_report`.

- [ ] **Step 3: Add orchestration functions**

Append to `scripts/benchmark/visualize_zero_shot_robustness.py`:

```python
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


def collect_domain_samples(
    *,
    dataset_root: Path,
    sample_per_class: int,
    seed: int,
) -> List[DatasetSample]:
    return collect_samples(
        dataset_root=dataset_root,
        limit_per_class=None,
        sample_per_class=sample_per_class,
        seed=seed,
    )


def run_report(
    *,
    datasets: Dict[str, str],
    decision_profiles: Dict[str, str],
    sample_per_class: int,
    robustness_sample_per_class: int,
    perturbation_names: Sequence[str],
    output_dir: Path,
    client,
    semantic_threshold: float,
    seed: int,
    workers: int,
    skip_robustness: bool,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    domain_suites: List[Dict] = []
    perturbation_suites: List[Dict] = []
    records_by_suite: Dict[str, List[PredictionRecord]] = {}
    clean_by_domain: Dict[str, Dict] = {}

    for index, (domain, root_str) in enumerate(datasets.items()):
        dataset_root = resolve_project_path(root_str)
        if not dataset_root.exists():
            warnings.append(f"Dataset root does not exist for {domain}: {dataset_root}")
            continue
        try:
            samples = collect_domain_samples(
                dataset_root=dataset_root,
                sample_per_class=sample_per_class,
                seed=seed + index,
            )
        except Exception as exc:
            warnings.append(f"Failed to collect samples for {domain}: {exc}")
            continue
        profile = decision_profiles.get(domain, "")
        suite_key = f"clean__{normalize_domain_name(domain)}"
        suite, records = run_suite(
            domain=domain,
            dataset_root=dataset_root,
            suite_key=suite_key,
            suite_kind="clean",
            decision_profile=profile,
            samples=samples,
            client=client,
            semantic_threshold=semantic_threshold,
            workers=workers,
        )
        suite_dict = suite_to_dict(suite)
        domain_suites.append(suite_dict)
        clean_by_domain[domain] = suite_dict
        records_by_suite[suite_key] = records

    if not domain_suites:
        raise RuntimeError("No clean domain suite could be executed.")

    if not skip_robustness:
        perturbations = build_perturbations(perturbation_names)
        with tempfile.TemporaryDirectory(prefix="zero_shot_robustness_") as temp_dir:
            workspace = Path(temp_dir)
            for domain_index, (domain, root_str) in enumerate(datasets.items()):
                clean_suite = clean_by_domain.get(domain)
                if clean_suite is None:
                    continue
                dataset_root = resolve_project_path(root_str)
                try:
                    base_samples = collect_domain_samples(
                        dataset_root=dataset_root,
                        sample_per_class=robustness_sample_per_class,
                        seed=seed + 1000 + domain_index,
                    )
                except Exception as exc:
                    warnings.append(f"Failed to collect robustness samples for {domain}: {exc}")
                    continue
                profile = decision_profiles.get(domain, "")
                for perturbation_index, perturbation in enumerate(perturbations):
                    perturbed_samples = materialize_perturbed_samples(
                        samples=base_samples,
                        perturbation=perturbation,
                        workspace=workspace / normalize_domain_name(domain),
                        seed=seed + 2000 + domain_index * 100 + perturbation_index,
                    )
                    suite_key = f"perturbed__{normalize_domain_name(domain)}__{perturbation.name}"
                    suite, records = run_suite(
                        domain=domain,
                        dataset_root=dataset_root,
                        suite_key=suite_key,
                        suite_kind="perturbed",
                        decision_profile=profile,
                        samples=perturbed_samples,
                        client=client,
                        semantic_threshold=semantic_threshold,
                        workers=workers,
                        perturbation=perturbation.name,
                    )
                    suite_dict = suite_to_dict(suite)
                    suite_dict["clean_reference"] = clean_suite["metrics"]
                    suite_dict["drops"] = compute_metric_drops(
                        clean_metrics=clean_suite["metrics"],
                        perturbed_metrics=suite_dict["metrics"],
                        clean_audit=clean_suite["audit_summary"],
                        perturbed_audit=suite_dict["audit_summary"],
                    )
                    perturbation_suites.append(suite_dict)
                    records_by_suite[suite_key] = records

    payload = build_report_payload(
        config={
            "datasets": dict(datasets),
            "decision_profiles": dict(decision_profiles),
            "sample_per_class": sample_per_class,
            "robustness_sample_per_class": robustness_sample_per_class,
            "perturbations": list(perturbation_names),
            "semantic_threshold": semantic_threshold,
            "seed": seed,
            "workers": workers,
            "skip_robustness": skip_robustness,
        },
        domain_suites=domain_suites,
        perturbation_suites=perturbation_suites,
        warnings=warnings,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_predictions_csv(records_by_suite, output_dir / "predictions.csv")
    (output_dir / "index.html").write_text(render_html_report(payload), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown_summary(payload), encoding="utf-8")
    return payload
```

- [ ] **Step 4: Add CLI parsing and main**

Append to `scripts/benchmark/visualize_zero_shot_robustness.py`:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zero-shot cross-domain and black-box perturbation robustness reports.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "Test=Datasets/Test",
            "Celeb-DF=Datasets/Celeb-DF",
            "DFDC=Datasets/DFDC",
            "WildDeepfake=Datasets/WildDeepfake",
        ],
        help="Domain=path entries. Example: DFDC=Datasets/DFDC",
    )
    parser.add_argument(
        "--decision-profiles",
        nargs="*",
        default=[
            "Celeb-DF=celeb_df",
            "DFDC=dfdc",
            "WildDeepfake=wilddeepfake",
        ],
        help="Optional Domain=profile entries used for threshold calibration.",
    )
    parser.add_argument("--sample-per-class", type=int, default=120)
    parser.add_argument("--robustness-sample-per-class", type=int, default=40)
    parser.add_argument(
        "--perturbations",
        nargs="*",
        default=[
            "jpeg_q60",
            "gaussian_blur",
            "downscale_restore",
            "crop_restore",
            "color_shift",
            "light_noise",
            "occlusion",
        ],
    )
    parser.add_argument("--output-dir", default="reports/report_zero_shot_robustness")
    parser.add_argument("--mode", choices=("internal", "http"), default="internal")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/detect")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--semantic-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-robustness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_key_value_items(args.datasets)
    decision_profiles = parse_key_value_items(args.decision_profiles)
    output_dir = resolve_project_path(args.output_dir)
    client = build_client(args.mode, args.endpoint, args.timeout)
    print(f"[RELOAD] zero-shot robustness report -> {output_dir}")
    print(f"[RELOAD] datasets: {', '.join(datasets)}")
    print(f"[RELOAD] perturbations: {', '.join(args.perturbations)}")
    payload = run_report(
        datasets=datasets,
        decision_profiles=decision_profiles,
        sample_per_class=args.sample_per_class,
        robustness_sample_per_class=args.robustness_sample_per_class,
        perturbation_names=args.perturbations,
        output_dir=output_dir,
        client=client,
        semantic_threshold=args.semantic_threshold,
        seed=args.seed,
        workers=args.workers,
        skip_robustness=args.skip_robustness,
    )
    summary = payload["summary"]
    print(
        "[RELOAD] summary: "
        f"domains={summary['domain_count']}, "
        f"mean_clean={format_percent(float(summary['mean_clean_accuracy']))}, "
        f"mean_perturbed={format_percent(float(summary['mean_perturbed_accuracy']))}, "
        f"max_drop={format_percent(float(summary['max_accuracy_drop']))}"
    )
    print(f"[CREATE] report written: {output_dir / 'index.html'}")
    print(f"[CREATE] metrics written: {output_dir / 'metrics.json'}")
    print(f"[CREATE] markdown written: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
```

Expected: PASS, 10 tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark/visualize_zero_shot_robustness.py tests/test_zero_shot_robustness_report.py
git commit -m "feat: add zero-shot robustness report cli"
```

---

### Task 5: Run Local and Remote Verification

**Files:**
- Use: `scripts/benchmark/visualize_zero_shot_robustness.py`
- Use: `tests/test_zero_shot_robustness_report.py`
- Output on remote smoke: `reports/report_zero_shot_robustness_smoke/`

- [ ] **Step 1: Run local unit tests**

Run:

```bash
python -m pytest tests/test_zero_shot_robustness_report.py tests/test_benchmark_audit_summary.py -q
```

Expected: PASS. If local Python lacks pytest, record the exact `No module named pytest` error and run the same command in the remote `detector` conda environment.

- [ ] **Step 2: Run frontend build check without rewriting tracked CSS**

Run from PowerShell:

```powershell
$out = Join-Path $env:TEMP 'graph-fakedetector-tailwind-build-check.css'
npx tailwindcss -i ./frontend/src/input.css -o $out --minify
$code = $LASTEXITCODE
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
exit $code
```

Expected: exit code 0 and Tailwind reports `Done`.

- [ ] **Step 3: Run remote pytest in the activated conda environment**

Run:

```bash
ssh -p 49649 root@ae2836a105e54a59892c240731db2e15.region1.waas.aigate.cc \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate detector && cd /root/pycode/graph_detect && python -m pytest tests/test_zero_shot_robustness_report.py tests/test_benchmark_audit_summary.py -q"
```

Expected: PASS. If `pytest` is missing, record the exact error and do not claim tests passed.

- [ ] **Step 4: Run remote smoke report with small samples**

Run:

```bash
ssh -p 49649 root@ae2836a105e54a59892c240731db2e15.region1.waas.aigate.cc \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate detector && cd /root/pycode/graph_detect && python scripts/benchmark/visualize_zero_shot_robustness.py --datasets Test=Datasets/Test DFDC=Datasets/DFDC --sample-per-class 5 --robustness-sample-per-class 3 --decision-profiles DFDC=dfdc --perturbations jpeg_q60 crop_restore --output-dir reports/report_zero_shot_robustness_smoke --workers 1"
```

Expected:

- `reports/report_zero_shot_robustness_smoke/metrics.json` exists.
- `reports/report_zero_shot_robustness_smoke/predictions.csv` exists.
- `reports/report_zero_shot_robustness_smoke/index.html` exists.
- `reports/report_zero_shot_robustness_smoke/summary.md` exists.
- Command summary prints `domains=2`.

- [ ] **Step 5: Inspect remote smoke metrics shape**

Run:

```bash
ssh -p 49649 root@ae2836a105e54a59892c240731db2e15.region1.waas.aigate.cc \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate detector && cd /root/pycode/graph_detect && python -c \"import json;from pathlib import Path;p=Path('reports/report_zero_shot_robustness_smoke/metrics.json');data=json.loads(p.read_text(encoding='utf-8'));print(data['report_type']);print(data['summary']['domain_count']);print(len(data['domain_suites']));print(len(data['perturbation_suites']));print(bool(data['recommendations'] or data['domain_suites']))\""
```

Expected output includes:

```text
zero_shot_robustness
2
2
4
True
```

- [ ] **Step 6: Commit if remote smoke creates tracked report files intentionally**

Do not commit remote smoke report files unless the user asks to preserve smoke outputs locally. The implementation commit from Task 4 is the code deliverable.

---

### Task 6: Optional Full Report Run

**Files:**
- Use: `scripts/benchmark/visualize_zero_shot_robustness.py`
- Output on remote: `reports/report_zero_shot_robustness_2026-06-07/`

- [ ] **Step 1: Ask before the full run**

Ask the user before running this command because it can be slower than the smoke run:

```text
是否现在跑完整 zero-shot/扰动报告？建议参数为四个数据域 clean 每类 120 张，扰动每类 40 张，7 种扰动。
```

- [ ] **Step 2: Run full report only after approval**

Run:

```bash
ssh -p 49649 root@ae2836a105e54a59892c240731db2e15.region1.waas.aigate.cc \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate detector && cd /root/pycode/graph_detect && python scripts/benchmark/visualize_zero_shot_robustness.py --datasets Test=Datasets/Test Celeb-DF=Datasets/Celeb-DF DFDC=Datasets/DFDC WildDeepfake=Datasets/WildDeepfake --sample-per-class 120 --robustness-sample-per-class 40 --decision-profiles Celeb-DF=celeb_df DFDC=dfdc WildDeepfake=wilddeepfake --output-dir reports/report_zero_shot_robustness_2026-06-07 --workers 1"
```

Expected:

- `metrics.json`, `predictions.csv`, `index.html`, `summary.md` are written.
- Summary prints `domains=4`.
- `summary.md` contains conservative language about engineering sampling and black-box approximate perturbations.

- [ ] **Step 3: Sync full report locally only after approval**

If the user wants the generated report locally, run:

```powershell
scp -P 49649 -r root@ae2836a105e54a59892c240731db2e15.region1.waas.aigate.cc:/root/pycode/graph_detect/reports/report_zero_shot_robustness_2026-06-07 .\reports\
```

Expected: local `reports/report_zero_shot_robustness_2026-06-07/` contains the four output files.
