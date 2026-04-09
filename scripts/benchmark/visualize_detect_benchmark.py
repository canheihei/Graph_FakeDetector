from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import html
import json
import math
import random
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib import error, request


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_LABELS = {"FAKE", "REAL"}
EXCLUDED_LABELS = {"OUT_OF_SCOPE", "NON_PORTRAIT"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    truth_label: str


@dataclass
class PredictionRecord:
    path: str
    file_name: str
    truth_label: str
    predicted_label: str
    confidence: float
    is_correct: bool
    latency_ms: float
    error: str = ""


@dataclass(frozen=True)
class BenchmarkSummary:
    total_samples: int
    excluded_predictions: int
    valid_predictions: int
    error_count: int
    correct_predictions: int
    accuracy_all: float
    accuracy_valid: float
    precision_fake: float
    recall_fake: float
    specificity_real: float
    f1_fake: float
    balanced_accuracy: float
    avg_confidence_correct: float
    avg_confidence_incorrect: float
    tp: int
    tn: int
    fp: int
    fn: int
    fake_support: int
    real_support: int


class DetectClient:
    def predict(self, image_path: Path, semantic_threshold: float) -> Dict:
        raise NotImplementedError


class InternalDetectClient(DetectClient):
    def __init__(self) -> None:
        from app import detect_facade
        from service.facades import DetectRequest

        self._detect_facade = detect_facade
        self._request_cls = DetectRequest

    def predict(self, image_path: Path, semantic_threshold: float) -> Dict:
        image_bytes = image_path.read_bytes()
        return self._detect_facade.execute(
            self._request_cls(
                image_bytes=image_bytes,
                auto_evolve_enabled=False,
                semantic_threshold=semantic_threshold,
                use_llm_generation=False,
            )
        )


class HttpDetectClient(DetectClient):
    def __init__(self, endpoint: str, timeout: float) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def predict(self, image_path: Path, semantic_threshold: float) -> Dict:
        payload, content_type = self._build_multipart_body(
            image_path=image_path,
            fields={
                "auto_evolve": "false",
                "use_llm_generation": "false",
                "semantic_threshold": str(semantic_threshold),
            },
        )
        req = request.Request(
            self._endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": content_type},
        )
        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"HTTP request failed: {exc.reason}") from exc

    @staticmethod
    def _build_multipart_body(image_path: Path, fields: Dict[str, str]) -> tuple[bytes, str]:
        boundary = f"----GraphDetectBoundary{uuid.uuid4().hex}"
        body = bytearray()

        for key, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
            )
            body.extend(f"{value}\r\n".encode("utf-8"))

        mime_type = "application/octet-stream"
        suffix = image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif suffix == ".png":
            mime_type = "image/png"
        elif suffix == ".bmp":
            mime_type = "image/bmp"
        elif suffix == ".webp":
            mime_type = "image/webp"

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.extend(image_path.read_bytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        return bytes(body), f"multipart/form-data; boundary={boundary}"


class DetectBenchmarkRunner:
    def __init__(
        self,
        client: DetectClient,
        semantic_threshold: float,
        workers: int = 1,
    ) -> None:
        self._client = client
        self._semantic_threshold = semantic_threshold
        self._workers = max(1, workers)

    def run(self, samples: Iterable[DatasetSample]) -> List[PredictionRecord]:
        sample_list = list(samples)
        total = len(sample_list)
        if total == 0:
            return []

        if self._workers == 1:
            return self._run_sequential(sample_list)
        return self._run_parallel(sample_list)

    def _run_sequential(self, sample_list: List[DatasetSample]) -> List[PredictionRecord]:
        records: List[PredictionRecord] = []
        total = len(sample_list)
        for index, sample in enumerate(sample_list, start=1):
            records.append(self._predict_one(sample))
            self._maybe_log_progress(index, total)
        return records

    def _run_parallel(self, sample_list: List[DatasetSample]) -> List[PredictionRecord]:
        total = len(sample_list)
        records_by_index: Dict[int, PredictionRecord] = {}
        completed = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            future_map = {
                executor.submit(self._predict_one, sample): index
                for index, sample in enumerate(sample_list)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                records_by_index[index] = future.result()
                with lock:
                    completed += 1
                    self._maybe_log_progress(completed, total)

        return [records_by_index[index] for index in range(total)]

    def _predict_one(self, sample: DatasetSample) -> PredictionRecord:
        start = time.perf_counter()
        try:
            result = self._client.predict(sample.path, self._semantic_threshold)
            predicted_label = normalize_label(result.get("label", "ERROR"))
            confidence = coerce_float(result.get("confidence", 0.0))
            error_message = ""
        except Exception as exc:
            predicted_label = "ERROR"
            confidence = 0.0
            error_message = str(exc)
            print(f"[WARN] detect failed: {sample.path.name} -> {error_message}")

        latency_ms = (time.perf_counter() - start) * 1000.0
        return PredictionRecord(
            path=str(sample.path),
            file_name=sample.path.name,
            truth_label=sample.truth_label,
            predicted_label=predicted_label,
            confidence=round(confidence, 6),
            is_correct=predicted_label == sample.truth_label,
            latency_ms=round(latency_ms, 3),
            error=error_message,
        )

    @staticmethod
    def _maybe_log_progress(index: int, total: int) -> None:
        if index == 1 or index % 50 == 0 or index == total:
            print(f"[RELOAD] progress: {index}/{total}")


def normalize_label(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in VALID_LABELS or text in EXCLUDED_LABELS:
        return text
    return "ERROR"


def coerce_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def resolve_class_dir(dataset_root: Path, aliases: Iterable[str]) -> Path:
    alias_map = {alias.lower(): alias for alias in aliases}
    existing_dirs = [path for path in dataset_root.iterdir() if path.is_dir()]

    for path in existing_dirs:
        if path.name.lower() in alias_map:
            return path

    for alias in aliases:
        direct = dataset_root / alias
        if direct.exists() and direct.is_dir():
            return direct

    discovered = ", ".join(sorted(path.name for path in existing_dirs)) or "<none>"
    expected = ", ".join(aliases)
    raise FileNotFoundError(
        f"Missing class directory for [{expected}] under {dataset_root}. "
        f"Discovered subdirectories: {discovered}"
    )


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


def collect_samples(
    dataset_root: Path,
    limit_per_class: Optional[int],
    sample_per_class: Optional[int],
    seed: int,
) -> List[DatasetSample]:
    samples: List[DatasetSample] = []
    rng = random.Random(seed)
    class_specs = [
        (("Fake", "fake", "FAKE", "fakes", "synthetic"), "FAKE"),
        (("Real", "real", "REAL", "reals", "authentic"), "REAL"),
    ]

    for aliases, truth_label in class_specs:
        class_dir = resolve_class_dir(dataset_root, aliases)

        class_files = sorted(
            path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if sample_per_class is not None and len(class_files) > sample_per_class:
            class_files = sorted(rng.sample(class_files, sample_per_class))
        if limit_per_class is not None:
            class_files = class_files[:limit_per_class]

        samples.extend(DatasetSample(path=path, truth_label=truth_label) for path in class_files)

    if not samples:
        raise RuntimeError(f"No images found under {dataset_root}")
    return samples


def compute_summary(records: List[PredictionRecord]) -> BenchmarkSummary:
    total = len(records)
    valid = [record for record in records if record.predicted_label in VALID_LABELS]
    excluded = [
        record for record in records if record.predicted_label in EXCLUDED_LABELS
    ]
    errors = [
        record
        for record in records
        if record.predicted_label not in VALID_LABELS
        and record.predicted_label not in EXCLUDED_LABELS
    ]
    correct = [record for record in records if record.is_correct]
    incorrect = [record for record in valid if not record.is_correct]

    tp = sum(1 for record in valid if record.truth_label == "FAKE" and record.predicted_label == "FAKE")
    tn = sum(1 for record in valid if record.truth_label == "REAL" and record.predicted_label == "REAL")
    fp = sum(1 for record in valid if record.truth_label == "REAL" and record.predicted_label == "FAKE")
    fn = sum(1 for record in valid if record.truth_label == "FAKE" and record.predicted_label == "REAL")

    fake_support = sum(1 for record in records if record.truth_label == "FAKE")
    real_support = sum(1 for record in records if record.truth_label == "REAL")

    valid_count = len(valid)
    excluded_count = len(excluded)
    correct_count = len(correct)
    error_count = len(errors)
    evaluated_total = total - excluded_count
    accuracy_all = safe_div(correct_count, evaluated_total)
    accuracy_valid = safe_div(tp + tn, valid_count)
    precision_fake = safe_div(tp, tp + fp)
    recall_fake = safe_div(tp, tp + fn)
    specificity_real = safe_div(tn, tn + fp)
    f1_fake = safe_div(2 * precision_fake * recall_fake, precision_fake + recall_fake)
    balanced_accuracy = (recall_fake + specificity_real) / 2.0

    avg_confidence_correct = average(
        record.confidence for record in valid if record.is_correct
    )
    avg_confidence_incorrect = average(
        record.confidence for record in incorrect
    )

    return BenchmarkSummary(
        total_samples=total,
        excluded_predictions=excluded_count,
        valid_predictions=valid_count,
        error_count=error_count,
        correct_predictions=correct_count,
        accuracy_all=accuracy_all,
        accuracy_valid=accuracy_valid,
        precision_fake=precision_fake,
        recall_fake=recall_fake,
        specificity_real=specificity_real,
        f1_fake=f1_fake,
        balanced_accuracy=balanced_accuracy,
        avg_confidence_correct=avg_confidence_correct,
        avg_confidence_incorrect=avg_confidence_incorrect,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        fake_support=fake_support,
        real_support=real_support,
    )


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def average(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def build_histogram(records: List[PredictionRecord], *, only_valid: bool, match: Optional[bool]) -> List[Dict]:
    bins = [0] * 10
    filtered = records
    if only_valid:
        filtered = [record for record in filtered if record.predicted_label in VALID_LABELS]
    if match is not None:
        filtered = [record for record in filtered if record.is_correct is match]

    for record in filtered:
        confidence = min(max(record.confidence, 0.0), 1.0)
        index = min(int(math.floor(confidence * 10.0)), 9)
        bins[index] += 1

    result = []
    for idx, count in enumerate(bins):
        lower = idx / 10.0
        upper = (idx + 1) / 10.0
        result.append(
            {
                "label": f"{lower:.1f}-{upper:.1f}",
                "count": count,
            }
        )
    return result


def build_top_errors(records: List[PredictionRecord], limit: int = 50) -> List[PredictionRecord]:
    errors = [record for record in records if not record.is_correct]
    return sorted(
        errors,
        key=lambda item: (
            0 if item.error else 1,
            -item.confidence,
            item.file_name,
        ),
    )[:limit]


def render_html_report(
    dataset_root: Path,
    mode: str,
    records: List[PredictionRecord],
    summary: BenchmarkSummary,
) -> str:
    correct_hist = build_histogram(records, only_valid=True, match=True)
    incorrect_hist = build_histogram(records, only_valid=True, match=False)
    top_errors = build_top_errors(records)

    cards_html = "".join(
        [
            render_card("Total Samples", str(summary.total_samples)),
            render_card("Accuracy (All)", format_percent(summary.accuracy_all)),
            render_card("Valid Coverage", format_percent(safe_div(summary.valid_predictions, summary.total_samples))),
            render_card("Errors", str(summary.error_count)),
            render_card("F1 (Fake)", format_percent(summary.f1_fake)),
            render_card("Balanced Accuracy", format_percent(summary.balanced_accuracy)),
        ]
    )

    metric_rows = "".join(
        render_metric_row(name, value)
        for name, value in [
            ("Accuracy (all)", format_percent(summary.accuracy_all)),
            ("Accuracy (valid only)", format_percent(summary.accuracy_valid)),
            ("Precision (Fake)", format_percent(summary.precision_fake)),
            ("Recall (Fake)", format_percent(summary.recall_fake)),
            ("Specificity (Real)", format_percent(summary.specificity_real)),
            ("F1 (Fake)", format_percent(summary.f1_fake)),
            ("Balanced Accuracy", format_percent(summary.balanced_accuracy)),
            ("Average confidence / correct", f"{summary.avg_confidence_correct:.3f}"),
            ("Average confidence / incorrect", f"{summary.avg_confidence_incorrect:.3f}"),
            ("Valid predictions", str(summary.valid_predictions)),
            ("Prediction errors", str(summary.error_count)),
        ]
    )

    class_rows = "".join(
        render_metric_row(name, value)
        for name, value in [
            ("Fake support", str(summary.fake_support)),
            ("Real support", str(summary.real_support)),
            ("True Positive", str(summary.tp)),
            ("True Negative", str(summary.tn)),
            ("False Positive", str(summary.fp)),
            ("False Negative", str(summary.fn)),
        ]
    )

    confusion_html = f"""
    <table class="matrix">
      <thead>
        <tr><th></th><th>Pred FAKE</th><th>Pred REAL</th></tr>
      </thead>
      <tbody>
        <tr><th>True FAKE</th><td>{summary.tp}</td><td>{summary.fn}</td></tr>
        <tr><th>True REAL</th><td>{summary.fp}</td><td>{summary.tn}</td></tr>
      </tbody>
    </table>
    """

    error_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(Path(record.path).name)}</td>
          <td>{html.escape(record.truth_label)}</td>
          <td>{html.escape(record.predicted_label)}</td>
          <td>{record.confidence:.3f}</td>
          <td>{record.latency_ms:.1f}</td>
          <td>{html.escape(record.error or "-")}</td>
        </tr>
        """
        for record in top_errors
    ) or '<tr><td colspan="6">No misclassified samples</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Detect Benchmark Report</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #e5dccf;
      --accent: #b45309;
      --accent-soft: #f59e0b;
      --green: #15803d;
      --red: #b91c1c;
      --blue: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.14), transparent 24%),
        linear-gradient(180deg, #f9f5ee 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 0.8fr;
      gap: 16px;
      margin-bottom: 24px;
    }}
    .panel {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 40px rgba(30, 41, 59, 0.06);
      padding: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
    }}
    .badge {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #fff4db;
      color: #92400e;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .card {{
      padding: 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, #fffdf8, #fff7eb);
      border: 1px solid #eadfce;
    }}
    .card .k {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .card .v {{
      font-size: 28px;
      font-weight: 800;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .metric-table, .error-table, .matrix {{
      width: 100%;
      border-collapse: collapse;
    }}
    .metric-table td, .error-table td, .error-table th, .matrix td, .matrix th {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      font-size: 14px;
    }}
    .metric-table td:last-child, .matrix td {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .matrix th {{
      background: #fcf7ed;
    }}
    .bars {{
      display: grid;
      gap: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 70px 1fr 50px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }}
    .bar-track {{
      height: 10px;
      border-radius: 999px;
      background: #f1e7d8;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent-soft), var(--accent));
    }}
    .bar-fill.bad {{
      background: linear-gradient(90deg, #fb7185, var(--red));
    }}
    .section-title {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .section-sub {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      display: inline-block;
      margin-right: 8px;
      margin-top: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f8efe2;
      font-size: 12px;
    }}
    .error-table th {{
      background: #fcf7ed;
    }}
    .footnote {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 960px) {{
      .hero, .grid, .cards {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <section class="panel">
        <div class="badge">Detect Benchmark</div>
        <h1>dataset/test detection report</h1>
        <p>Dataset: <strong>{html.escape(str(dataset_root))}</strong></p>
        <p>Mode: <strong>{html.escape(mode)}</strong></p>
        <p>This report summarizes fake-vs-real predictions, confusion counts, key metrics, and hard failure cases.</p>
      </section>
      <section class="panel">
        <h2 class="section-title">Run Summary</h2>
        <div class="pill">Total {summary.total_samples}</div>
        <div class="pill">Valid {summary.valid_predictions}</div>
        <div class="pill">Errors {summary.error_count}</div>
        <div class="pill">Correct {summary.correct_predictions}</div>
        <div class="footnote">
          Accuracy(all) counts request failures in the denominator. Accuracy(valid only) uses only valid FAKE/REAL outputs.
        </div>
      </section>
    </div>

    <section class="cards">
      {cards_html}
    </section>

    <section class="grid">
      <div class="panel">
        <h2 class="section-title">Core Metrics</h2>
        <table class="metric-table">
          <tbody>{metric_rows}</tbody>
        </table>
      </div>
      <div class="panel">
        <h2 class="section-title">Confusion Matrix</h2>
        <p class="section-sub">Only samples with valid FAKE/REAL predictions are counted here.</p>
        {confusion_html}
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2 class="section-title">Class Support</h2>
        <table class="metric-table">
          <tbody>{class_rows}</tbody>
        </table>
      </div>
      <div class="panel">
        <h2 class="section-title">Confidence Histogram</h2>
        <p class="section-sub">Binned by 0.1 confidence intervals, split by correct vs incorrect predictions.</p>
        <h3 class="section-sub">Correct Predictions</h3>
        {render_histogram_bars(correct_hist, bad=False)}
        <h3 class="section-sub" style="margin-top:16px;">Incorrect Predictions</h3>
        {render_histogram_bars(incorrect_hist, bad=True)}
      </div>
    </section>

    <section class="panel">
      <h2 class="section-title">Top Misclassified Samples</h2>
      <p class="section-sub">Rows are ordered by request failure first, then by high-confidence mistakes.</p>
      <table class="error-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Truth</th>
            <th>Pred</th>
            <th>Confidence</th>
            <th>Latency(ms)</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody>{error_rows}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def render_card(key: str, value: str) -> str:
    return f"""
    <div class="card">
      <div class="k">{html.escape(key)}</div>
      <div class="v">{html.escape(value)}</div>
    </div>
    """


def render_metric_row(name: str, value: str) -> str:
    return (
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>"
    )


def render_histogram_bars(items: List[Dict], *, bad: bool) -> str:
    max_count = max((item["count"] for item in items), default=1)
    rows = []
    for item in items:
        width = safe_div(item["count"], max_count) * 100.0 if max_count else 0.0
        rows.append(
            f"""
            <div class="bar-row">
              <span>{html.escape(item['label'])}</span>
              <div class="bar-track">
                <div class="bar-fill{' bad' if bad else ''}" style="width:{width:.2f}%"></div>
              </div>
              <span>{item['count']}</span>
            </div>
            """
        )
    return f'<div class="bars">{"".join(rows)}</div>'


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_csv(records: List[PredictionRecord], target: Path) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
        for record in records:
            writer.writerow(asdict(record))


def write_json(summary: BenchmarkSummary, records: List[PredictionRecord], target: Path) -> None:
    payload = {
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the /detect module on dataset/test and generate an HTML report.",
    )
    parser.add_argument(
        "--dataset-root",
        default="Datasets/Test",
        help="Dataset root containing Fake/ and Real/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/detect_benchmark",
        help="Directory used to write HTML/CSV/JSON reports.",
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
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap on the number of Fake/Real samples.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Randomly sample N images per class before evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --sample-per-class.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of benchmark worker threads. Default keeps sequential behavior.",
    )
    return parser.parse_args()


def make_client(args: argparse.Namespace) -> DetectClient:
    if args.mode == "http":
        return HttpDetectClient(endpoint=args.endpoint, timeout=args.timeout)
    return InternalDetectClient()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_project_path(args.dataset_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RELOAD] scanning dataset: {dataset_root}")
    samples = collect_samples(
        dataset_root,
        args.limit_per_class,
        args.sample_per_class,
        args.seed,
    )
    print(f"[CREATE] collected {len(samples)} samples")

    client = make_client(args)
    runner = DetectBenchmarkRunner(
        client=client,
        semantic_threshold=args.semantic_threshold,
        workers=args.workers,
    )
    records = runner.run(samples)
    summary = compute_summary(records)

    html_report = render_html_report(
        dataset_root=dataset_root,
        mode=args.mode,
        records=records,
        summary=summary,
    )

    html_path = output_dir / "index.html"
    csv_path = output_dir / "predictions.csv"
    json_path = output_dir / "metrics.json"

    html_path.write_text(html_report, encoding="utf-8")
    write_csv(records, csv_path)
    write_json(summary, records, json_path)

    print(f"[CREATE] report written: {html_path}")
    print(f"[CREATE] csv written: {csv_path}")
    print(f"[CREATE] json written: {json_path}")
    print(
        "[RELOAD] summary: "
        f"accuracy_all={format_percent(summary.accuracy_all)}, "
        f"valid_coverage={format_percent(safe_div(summary.valid_predictions, summary.total_samples))}, "
        f"errors={summary.error_count}"
    )


if __name__ == "__main__":
    main()
