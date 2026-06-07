from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import APP_ROOT, candidate_review_facade, detect_facade
from scripts.benchmark.visualize_detect_benchmark import (
    collect_samples,
    compute_audit_summary,
    compute_summary,
)
from service.candidate_benchmark import _result_to_prediction_record
from service.candidate_generation import generate_candidate_items, should_generate_candidates
from service.candidate_review import (
    CandidateBenchmarkRequest,
    CandidateDeleteRequest,
    CandidatePromoteRequest,
    CandidateRequest,
)
from service.facades import DetectRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a same-domain small-sample candidate promote smoke test and compare evidence metrics before/after.",
    )
    parser.add_argument(
        "--dataset-root",
        default="Datasets/Test",
        help="Dataset root used for baseline/promoted comparison.",
    )
    parser.add_argument(
        "--scan-fake-limit",
        type=int,
        default=30,
        help="How many fake samples to scan to find a candidate-eligible sample.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=10,
        help="How many samples per class to use in the before/after comparison.",
    )
    parser.add_argument(
        "--benchmark-sample-per-class",
        type=int,
        default=10,
        help="How many samples per class to use in candidate benchmark.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.80,
        help="Semantic threshold forwarded to detect/benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--output",
        default="reports/report_candidate_promotion_same_domain_smoke.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw.resolve()
    return (APP_ROOT / raw).resolve()


def run_detect(image_path: Path, semantic_threshold: float) -> Dict[str, Any]:
    return detect_facade.execute(
        DetectRequest(
            image_bytes=image_path.read_bytes(),
            auto_evolve_enabled=False,
            semantic_threshold=semantic_threshold,
            use_llm_generation=False,
            decision_profile=None,
            decision_threshold_override=None,
        )
    )


def evaluate_samples(samples, semantic_threshold: float) -> Dict[str, Any]:
    records = []
    for sample in samples:
        response = run_detect(sample.path, semantic_threshold)
        records.append(_result_to_prediction_record(sample.path, sample.truth_label, response))
    summary = compute_summary(records)
    audit_summary = compute_audit_summary(records)
    return {
        "summary": asdict(summary),
        "audit_summary": audit_summary,
    }


def compute_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    before_audit = dict(before.get("audit_summary", {}) or {})
    after_audit = dict(after.get("audit_summary", {}) or {})
    keys = [
        "evidence_hit_rate",
        "evidence_hit_rate_valid",
        "fake_evidence_hit_rate",
        "high_score_no_evidence_rate",
        "unresolved_subdomain_rate",
        "avg_evidence_alignment_score",
        "joint_evidence_correct_rate",
        "fake_joint_evidence_recall",
    ]
    return {
        key: float(after_audit.get(key, 0.0) or 0.0) - float(before_audit.get(key, 0.0) or 0.0)
        for key in keys
    }


def backup_file(path: Path, suffix: str) -> str | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
    shutil.copyfile(path, backup)
    return str(backup)


def find_candidate_eligible_sample(fake_root: Path, limit: int, semantic_threshold: float) -> Dict[str, Any]:
    fake_files = sorted(path for path in fake_root.iterdir() if path.is_file())[:limit]
    attempts = []
    fallback_context: Dict[str, Any] | None = None
    for file_path in fake_files:
        detect_result = run_detect(file_path, semantic_threshold)
        attempt = {
            "file_name": file_path.name,
            "label": detect_result.get("label"),
            "reasoning_type": detect_result.get("reasoning_type"),
            "decision_fake_score": detect_result.get("decision_fake_score"),
            "evidence_count": len(detect_result.get("evidence", []) or []),
            "eligible": should_generate_candidates(detect_result),
        }
        attempts.append(attempt)
        if fallback_context is None and str(detect_result.get("label", "")).strip().upper() == "FAKE":
            fallback_context = {
                "sample_path": str(file_path),
                "detect_result": detect_result,
            }
        if attempt["eligible"]:
            return {
                "sample_path": str(file_path),
                "detect_result": detect_result,
                "attempts": attempts,
                "forced_generation": False,
            }
    if fallback_context is not None:
        return {
            **fallback_context,
            "attempts": attempts,
            "forced_generation": True,
        }
    raise RuntimeError("No fake sample found in scanned same-domain subset")


def main() -> None:
    args = parse_args()
    dataset_root = resolve_project_path(args.dataset_root)
    fake_root = dataset_root / "Fake"
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    mapping_backup = backup_file(APP_ROOT / "alignment" / "mapping_config.json", f"smoke_backup_{timestamp}")
    candidate_backup = backup_file(APP_ROOT / "alignment" / "mapping_candidates.json", f"smoke_backup_{timestamp}")

    candidate_context = find_candidate_eligible_sample(
        fake_root=fake_root,
        limit=args.scan_fake_limit,
        semantic_threshold=args.semantic_threshold,
    )
    sample_path = Path(candidate_context["sample_path"])
    detect_result = candidate_context["detect_result"]
    forced_generation = bool(candidate_context.get("forced_generation", False))

    if forced_generation:
        generated_items = generate_candidate_items(
            detect_result=detect_result,
            source_sample_name=sample_path.name,
            decision_profile=None,
        )
        candidate_review_facade._candidate_store.append_items(generated_items)
        candidate_review_facade._candidate_graph_store.persist_candidates(generated_items)
    else:
        generated = candidate_review_facade.generate(
            CandidateRequest(
                detect_result=detect_result,
                source_sample_name=sample_path.name,
                decision_profile=None,
            )
        )
        generated_items = list(generated.get("items", []) or [])
    if not generated_items:
        raise RuntimeError("Candidate generation returned no items")
    selected_candidate = generated_items[0]
    selected_candidate_id = str(selected_candidate.get("candidate_id"))

    compare_samples = collect_samples(
        dataset_root=dataset_root,
        limit_per_class=None,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    before_metrics = evaluate_samples(compare_samples, args.semantic_threshold)

    benchmark_payload = candidate_review_facade.benchmark(
        CandidateBenchmarkRequest(
            candidate_ids=[selected_candidate_id],
            mode="quick",
            decision_profile=None,
            sample_per_class=args.benchmark_sample_per_class,
            semantic_threshold=args.semantic_threshold,
            decision_threshold_override=None,
        )
    )
    benchmark_result = dict(benchmark_payload.get("result", {}) or {})
    if not benchmark_result.get("passed", False):
        candidate_review_facade.delete(CandidateDeleteRequest(candidate_ids=[item.get("candidate_id") for item in generated_items]))
        raise RuntimeError("Selected candidate did not pass quick benchmark gating")

    promote_payload = candidate_review_facade.promote(
        CandidatePromoteRequest(candidate_ids=[selected_candidate_id])
    )
    after_metrics = evaluate_samples(compare_samples, args.semantic_threshold)

    payload = {
        "timestamp_utc": timestamp,
        "dataset_root": str(dataset_root),
        "sample_per_class": args.sample_per_class,
        "benchmark_sample_per_class": args.benchmark_sample_per_class,
        "semantic_threshold": args.semantic_threshold,
        "seed": args.seed,
        "backups": {
            "mapping_config": mapping_backup,
            "mapping_candidates": candidate_backup,
        },
        "candidate_source_sample": {
            "path": str(sample_path),
            "file_name": sample_path.name,
            "forced_generation": forced_generation,
            "reasoning_type": detect_result.get("reasoning_type"),
            "decision_fake_score": detect_result.get("decision_fake_score"),
            "evidence_count": len(detect_result.get("evidence", []) or []),
        },
        "scan_attempts": candidate_context["attempts"],
        "generated_candidate_count": len(generated_items),
        "selected_candidate": {
            "candidate_id": selected_candidate_id,
            "graph_candidate": selected_candidate.get("graph_candidate", {}),
            "mapping_candidate": selected_candidate.get("mapping_candidate", {}),
            "llm": selected_candidate.get("llm", {}),
        },
        "before": before_metrics,
        "benchmark_overlay": benchmark_result,
        "promote": promote_payload,
        "after": after_metrics,
        "delta": {
            "audit_summary": compute_delta(before_metrics, after_metrics),
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
