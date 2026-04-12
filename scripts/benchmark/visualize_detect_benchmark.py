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
    decision_fake_score: float = 0.0
    decision_threshold: float = 0.0
    decision_margin: float = 0.0
    score_source: str = ""
    threshold_source: str = ""
    decision_profile: str = ""
    reasoning_type: str = ""
    risk_level: str = ""
    needs_review: bool = False
    review_reasons_count: int = 0
    diagnostic_chain_len: int = 0
    evidence_count: int = 0
    evidence_requested: int = 0
    evidence_unresolved: int = 0
    evidence_alignment_score: float = 0.0
    graph_influence_weight: float = 0.0
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
    def predict(
        self,
        image_path: Path,
        semantic_threshold: float,
        decision_profile: Optional[str],
        decision_threshold_override: Optional[float],
    ) -> Dict:
        raise NotImplementedError


class InternalDetectClient(DetectClient):
    def __init__(self) -> None:
        from app import detect_facade
        from service.facades import DetectRequest

        self._detect_facade = detect_facade
        self._request_cls = DetectRequest

    def predict(
        self,
        image_path: Path,
        semantic_threshold: float,
        decision_profile: Optional[str],
        decision_threshold_override: Optional[float],
    ) -> Dict:
        image_bytes = image_path.read_bytes()
        return self._detect_facade.execute(
            self._request_cls(
                image_bytes=image_bytes,
                auto_evolve_enabled=False,
                semantic_threshold=semantic_threshold,
                use_llm_generation=False,
                decision_profile=decision_profile,
                decision_threshold_override=decision_threshold_override,
            )
        )


class HttpDetectClient(DetectClient):
    def __init__(self, endpoint: str, timeout: float) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def predict(
        self,
        image_path: Path,
        semantic_threshold: float,
        decision_profile: Optional[str],
        decision_threshold_override: Optional[float],
    ) -> Dict:
        fields = {
            "auto_evolve": "false",
            "use_llm_generation": "false",
            "semantic_threshold": str(semantic_threshold),
        }
        if decision_profile:
            fields["decision_profile"] = str(decision_profile)
        if decision_threshold_override is not None:
            fields["decision_threshold_override"] = str(decision_threshold_override)
        payload, content_type = self._build_multipart_body(
            image_path=image_path,
            fields=fields,
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
        decision_profile: Optional[str] = None,
        decision_threshold_override: Optional[float] = None,
        workers: int = 1,
    ) -> None:
        self._client = client
        self._semantic_threshold = semantic_threshold
        self._decision_profile = decision_profile
        self._decision_threshold_override = decision_threshold_override
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
            result = self._client.predict(
                sample.path,
                self._semantic_threshold,
                self._decision_profile,
                self._decision_threshold_override,
            )
            predicted_label = normalize_label(result.get("label", "ERROR"))
            confidence = coerce_float(result.get("confidence", 0.0))
            decision_fake_score = coerce_float(result.get("decision_fake_score", 0.0))
            decision_threshold = coerce_float(result.get("decision_threshold", 0.0))
            decision_margin = coerce_float(result.get("decision_margin", 0.0))
            score_source = str(result.get("score_source", "") or "")
            threshold_source = str(result.get("threshold_source", "") or "")
            decision_profile = str(result.get("decision_profile", "") or "")
            reasoning_type = str(result.get("reasoning_type", "") or "")
            risk_level = str(result.get("risk_level", "") or "")
            needs_review = bool(result.get("needs_review", False))
            review_reasons = result.get("review_reasons", [])
            review_reasons_count = len(review_reasons) if isinstance(review_reasons, list) else 0
            diagnostic_chain = result.get("diagnostic_chain", [])
            diagnostic_chain_len = len(diagnostic_chain) if isinstance(diagnostic_chain, list) else 0
            evidence = result.get("evidence", [])
            evidence_count = len(evidence) if isinstance(evidence, list) else 0
            evidence_diagnostics = result.get("evidence_diagnostics", {})
            if isinstance(evidence_diagnostics, dict):
                evidence_requested = int(evidence_diagnostics.get("requested_subdomains", 0) or 0)
                evidence_unresolved = int(evidence_diagnostics.get("unresolved_subdomains", 0) or 0)
            else:
                evidence_requested = 0
                evidence_unresolved = 0
            evidence_alignment_score = coerce_float(result.get("evidence_alignment_score", 0.0))
            graph_influence_weight = coerce_float(result.get("graph_influence_weight", 0.0))
            error_message = ""
        except Exception as exc:
            predicted_label = "ERROR"
            confidence = 0.0
            decision_fake_score = 0.0
            decision_threshold = 0.0
            decision_margin = 0.0
            score_source = ""
            threshold_source = ""
            decision_profile = ""
            reasoning_type = ""
            risk_level = ""
            needs_review = False
            review_reasons_count = 0
            diagnostic_chain_len = 0
            evidence_count = 0
            evidence_requested = 0
            evidence_unresolved = 0
            evidence_alignment_score = 0.0
            graph_influence_weight = 0.0
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
            decision_fake_score=round(decision_fake_score, 6),
            decision_threshold=round(decision_threshold, 6),
            decision_margin=round(decision_margin, 6),
            score_source=score_source,
            threshold_source=threshold_source,
            decision_profile=decision_profile,
            reasoning_type=reasoning_type,
            risk_level=risk_level,
            needs_review=needs_review,
            review_reasons_count=review_reasons_count,
            diagnostic_chain_len=diagnostic_chain_len,
            evidence_count=evidence_count,
            evidence_requested=evidence_requested,
            evidence_unresolved=evidence_unresolved,
            evidence_alignment_score=round(evidence_alignment_score, 6),
            graph_influence_weight=round(graph_influence_weight, 6),
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


def summarize_by_threshold(records: List[PredictionRecord], threshold: float) -> Dict[str, float | int]:
    valid_records = [record for record in records if record.predicted_label in VALID_LABELS]
    if not valid_records:
        return {
            "threshold": round(float(threshold), 4),
            "valid_count": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy_valid": 0.0,
            "balanced_accuracy": 0.0,
            "precision_fake": 0.0,
            "recall_fake": 0.0,
            "specificity_real": 0.0,
        }

    tp = tn = fp = fn = 0
    for record in valid_records:
        pred_fake = record.decision_fake_score >= threshold
        if record.truth_label == "FAKE" and pred_fake:
            tp += 1
        elif record.truth_label == "REAL" and not pred_fake:
            tn += 1
        elif record.truth_label == "REAL" and pred_fake:
            fp += 1
        else:
            fn += 1

    accuracy_valid = safe_div(tp + tn, len(valid_records))
    recall_fake = safe_div(tp, tp + fn)
    specificity_real = safe_div(tn, tn + fp)
    precision_fake = safe_div(tp, tp + fp)
    balanced_accuracy = 0.5 * (recall_fake + specificity_real)

    return {
        "threshold": round(float(threshold), 4),
        "valid_count": len(valid_records),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_valid": accuracy_valid,
        "balanced_accuracy": balanced_accuracy,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "specificity_real": specificity_real,
    }


def find_recommended_threshold(records: List[PredictionRecord]) -> Optional[Dict[str, float | int]]:
    valid_records = [record for record in records if record.predicted_label in VALID_LABELS]
    if not valid_records:
        return None

    best_result = None
    for index in range(5, 96):
        threshold = index / 100.0
        summary = summarize_by_threshold(valid_records, threshold)
        score = (
            float(summary["balanced_accuracy"]),
            float(summary["accuracy_valid"]),
            int(summary["tp"]) - int(summary["fp"]),
        )
        if best_result is None or score > best_result[0]:
            best_result = (score, summary)

    assert best_result is not None
    selected = dict(best_result[1])
    selected["average_current_threshold"] = average(
        record.decision_threshold for record in valid_records
    )
    selected["valid_count"] = len(valid_records)
    return selected


def compute_audit_summary(records: List[PredictionRecord]) -> Dict[str, object]:
    total = len(records)
    if total == 0:
        return {
            "total_records": 0,
            "reasoning_type_coverage": 0.0,
            "diagnostic_chain_coverage": 0.0,
            "needs_review_rate": 0.0,
            "risk_level_distribution": {},
            "reasoning_type_distribution": {},
            "avg_diagnostic_chain_len": 0.0,
            "evidence_hit_rate": 0.0,
            "evidence_hit_rate_valid": 0.0,
            "fake_evidence_hit_rate": 0.0,
            "high_score_no_evidence_rate": 0.0,
            "unresolved_subdomain_rate": 0.0,
            "avg_evidence_alignment_score": 0.0,
        }

    reasoning_type_count: Dict[str, int] = {}
    risk_level_count: Dict[str, int] = {}
    non_empty_reasoning = 0
    has_chain = 0
    needs_review_count = 0
    chain_len_sum = 0
    evidence_hit_count = 0
    valid_count = 0
    valid_evidence_hit_count = 0
    fake_total = 0
    fake_evidence_hit_count = 0
    high_score_no_evidence_count = 0
    evidence_requested_sum = 0
    evidence_unresolved_sum = 0
    evidence_alignment_sum = 0.0

    for record in records:
        if record.reasoning_type:
            non_empty_reasoning += 1
            reasoning_type_count[record.reasoning_type] = reasoning_type_count.get(record.reasoning_type, 0) + 1
        if record.risk_level:
            risk_level_count[record.risk_level] = risk_level_count.get(record.risk_level, 0) + 1
        if record.needs_review:
            needs_review_count += 1
        if record.diagnostic_chain_len > 0:
            has_chain += 1
        chain_len_sum += int(record.diagnostic_chain_len)
        if record.evidence_count > 0:
            evidence_hit_count += 1
        if record.predicted_label in VALID_LABELS:
            valid_count += 1
            if record.evidence_count > 0:
                valid_evidence_hit_count += 1
            if record.evidence_count <= 0 and record.decision_fake_score >= (record.decision_threshold + 0.08):
                high_score_no_evidence_count += 1
        if record.truth_label == "FAKE":
            fake_total += 1
            if record.evidence_count > 0:
                fake_evidence_hit_count += 1
        evidence_requested_sum += max(int(record.evidence_requested), 0)
        evidence_unresolved_sum += max(int(record.evidence_unresolved), 0)
        evidence_alignment_sum += max(min(float(record.evidence_alignment_score), 1.0), 0.0)

    return {
        "total_records": total,
        "reasoning_type_coverage": safe_div(non_empty_reasoning, total),
        "diagnostic_chain_coverage": safe_div(has_chain, total),
        "needs_review_rate": safe_div(needs_review_count, total),
        "risk_level_distribution": dict(sorted(risk_level_count.items())),
        "reasoning_type_distribution": dict(sorted(reasoning_type_count.items())),
        "avg_diagnostic_chain_len": safe_div(chain_len_sum, total),
        "evidence_hit_rate": safe_div(evidence_hit_count, total),
        "evidence_hit_rate_valid": safe_div(valid_evidence_hit_count, valid_count),
        "fake_evidence_hit_rate": safe_div(fake_evidence_hit_count, fake_total),
        "high_score_no_evidence_rate": safe_div(high_score_no_evidence_count, valid_count),
        "unresolved_subdomain_rate": safe_div(evidence_unresolved_sum, evidence_requested_sum),
        "avg_evidence_alignment_score": safe_div(evidence_alignment_sum, total),
    }


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
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_json(
    summary: BenchmarkSummary,
    records: List[PredictionRecord],
    target: Path,
    calibration: Optional[Dict[str, float | int]],
    audit_summary: Dict[str, object],
) -> None:
    payload = {
        "summary": asdict(summary),
        "threshold_calibration": calibration,
        "audit_summary": audit_summary,
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
    parser.add_argument(
        "--decision-profile",
        default=None,
        help="Optional domain profile name forwarded to /detect for threshold calibration (e.g. celeb_df, dfdc).",
    )
    parser.add_argument(
        "--decision-threshold-override",
        type=float,
        default=None,
        help="Optional explicit decision threshold override in [0,1]. Takes precedence over profile.",
    )
    return parser.parse_args()


def make_client(args: argparse.Namespace) -> DetectClient:
    if args.mode == "http":
        return HttpDetectClient(endpoint=args.endpoint, timeout=args.timeout)
    return InternalDetectClient()


def main() -> None:
    args = parse_args()
    if args.decision_threshold_override is not None and not 0.0 <= args.decision_threshold_override <= 1.0:
        raise ValueError("--decision-threshold-override must be between 0 and 1")
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
        decision_profile=args.decision_profile,
        decision_threshold_override=args.decision_threshold_override,
        workers=args.workers,
    )
    records = runner.run(samples)
    summary = compute_summary(records)
    calibration = find_recommended_threshold(records)
    audit_summary = compute_audit_summary(records)

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
    write_json(summary, records, json_path, calibration, audit_summary)

    print(f"[CREATE] report written: {html_path}")
    print(f"[CREATE] csv written: {csv_path}")
    print(f"[CREATE] json written: {json_path}")
    print(
        "[RELOAD] summary: "
        f"accuracy_all={format_percent(summary.accuracy_all)}, "
        f"valid_coverage={format_percent(safe_div(summary.valid_predictions, summary.total_samples))}, "
        f"errors={summary.error_count}"
    )
    if calibration:
        print(
            "[RELOAD] threshold calibration: "
            f"recommended={calibration['threshold']:.2f}, "
            f"balanced_acc={format_percent(float(calibration['balanced_accuracy']))}, "
            f"accuracy_valid={format_percent(float(calibration['accuracy_valid']))}"
        )
    print(
        "[RELOAD] audit coverage: "
        f"reasoning_type={format_percent(float(audit_summary['reasoning_type_coverage']))}, "
        f"diagnostic_chain={format_percent(float(audit_summary['diagnostic_chain_coverage']))}, "
        f"needs_review={format_percent(float(audit_summary['needs_review_rate']))}"
    )
    print(
        "[RELOAD] evidence hit: "
        f"overall={format_percent(float(audit_summary['evidence_hit_rate']))}, "
        f"valid={format_percent(float(audit_summary['evidence_hit_rate_valid']))}, "
        f"fake={format_percent(float(audit_summary['fake_evidence_hit_rate']))}, "
        f"high_score_no_evidence={format_percent(float(audit_summary['high_score_no_evidence_rate']))}, "
        f"unresolved={format_percent(float(audit_summary['unresolved_subdomain_rate']))}"
    )


if __name__ == "__main__":
    main()
