from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from alignment.aligner import FeatureOntologyAligner
from scripts.benchmark.visualize_detect_benchmark import (
    PredictionRecord,
    collect_samples,
    compute_audit_summary,
    compute_summary,
    find_recommended_threshold,
)
from service.facades import DetectRequest, DetectionFacade


def _candidate_rule_key(candidate: Dict[str, Any]) -> tuple[str, str]:
    mapping = candidate.get("mapping_candidate", {}) or {}
    detector = str(mapping.get("detector", "")).strip()
    feature = str(mapping.get("feature", "")).strip()
    return detector, feature


def ensure_no_candidate_conflicts(selected_candidates: Sequence[Dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for candidate in selected_candidates:
        key = _candidate_rule_key(candidate)
        if not all(key):
            raise ValueError("candidate mapping is missing detector or feature")
        candidate_id = str(candidate.get("candidate_id", ""))
        if key in seen:
            raise ValueError(
                "multiple candidates selected for the same detector/feature: "
                f"{key[0]}:{key[1]} ({seen[key]}, {candidate_id})"
            )
        seen[key] = candidate_id


def _normalize_candidate_rule(candidate: Dict[str, Any]) -> Dict[str, Any]:
    mapping = dict(candidate.get("mapping_candidate", {}) or {})
    graph = dict(candidate.get("graph_candidate", {}) or {})
    mapping.setdefault("subdomain_id", graph.get("candidate_subdomain_id") or graph.get("subdomain_id"))
    mapping.setdefault("subdomain_label", graph.get("subdomain_name"))
    mapping.setdefault("evidence_enabled", False)
    mapping.setdefault("sigmoid_k", 8.0)
    mapping.setdefault("sigmoid_x0", 0.5)
    mapping.setdefault("weight", 0.70)
    mapping.setdefault("activation_threshold", 0.58)
    mapping.setdefault("context_detector", "")
    mapping.setdefault("context_feature", "")
    mapping.setdefault("context_min_value", 0.0)
    return mapping


def merge_mapping_rules(active_config: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ensure_no_candidate_conflicts(candidates)
    rules: dict[tuple[str, str], Dict[str, Any]] = {}
    for rule in active_config.get("rules", []):
        detector = str(rule.get("detector", "")).strip()
        feature = str(rule.get("feature", "")).strip()
        if detector and feature:
            rules[(detector, feature)] = dict(rule)

    for candidate in candidates:
        normalized = _normalize_candidate_rule(candidate)
        key = (
            str(normalized.get("detector", "")).strip(),
            str(normalized.get("feature", "")).strip(),
        )
        rules[key] = normalized

    return {
        "version": active_config.get("version", "1.0"),
        "rules": list(rules.values()),
    }


def promote_candidate_rules(mapping_path: Path | str, selected_candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mapping_file = Path(mapping_path)
    active_config = json.loads(mapping_file.read_text(encoding="utf-8"))
    merged = merge_mapping_rules(active_config, selected_candidates)
    mapping_file.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return merged


def _result_to_prediction_record(path: Path, truth_label: str, response: Dict[str, Any]) -> PredictionRecord:
    evidence = response.get("evidence", [])
    evidence_diagnostics = response.get("evidence_diagnostics", {}) or {}
    review_reasons = response.get("review_reasons", [])
    diagnostic_chain = response.get("diagnostic_chain", [])
    predicted_label = str(response.get("label", "ERROR"))
    return PredictionRecord(
        path=str(path),
        file_name=path.name,
        truth_label=truth_label,
        predicted_label=predicted_label,
        confidence=float(response.get("confidence", 0.0) or 0.0),
        is_correct=predicted_label == truth_label,
        latency_ms=0.0,
        decision_fake_score=float(response.get("decision_fake_score", 0.0) or 0.0),
        decision_threshold=float(response.get("decision_threshold", 0.0) or 0.0),
        decision_margin=float(response.get("decision_margin", 0.0) or 0.0),
        score_source=str(response.get("score_source", "") or ""),
        threshold_source=str(response.get("threshold_source", "") or ""),
        decision_profile=str(response.get("decision_profile", "") or ""),
        reasoning_type=str(response.get("reasoning_type", "") or ""),
        risk_level=str(response.get("risk_level", "") or ""),
        needs_review=bool(response.get("needs_review", False)),
        review_reasons_count=len(review_reasons) if isinstance(review_reasons, list) else 0,
        diagnostic_chain_len=len(diagnostic_chain) if isinstance(diagnostic_chain, list) else 0,
        evidence_count=len(evidence) if isinstance(evidence, list) else 0,
        evidence_requested=int(evidence_diagnostics.get("requested_subdomains", 0) or 0),
        evidence_unresolved=int(evidence_diagnostics.get("unresolved_subdomains", 0) or 0),
        evidence_alignment_score=float(response.get("evidence_alignment_score", 0.0) or 0.0),
        graph_influence_weight=float(response.get("graph_influence_weight", 0.0) or 0.0),
        error="",
    )


class CandidateBenchmarkRunner:
    def __init__(
        self,
        *,
        hub,
        graph_evolver,
        evidence_builder,
        logger,
        dataset_profile_roots: Dict[str, str],
        active_mapping_path: Path | str,
    ) -> None:
        self._hub = hub
        self._graph_evolver = graph_evolver
        self._evidence_builder = evidence_builder
        self._logger = logger
        self._dataset_profile_roots = dict(dataset_profile_roots)
        self._active_mapping_path = Path(active_mapping_path)

    def run(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        decision_profile: str | None,
        sample_per_class: int,
        semantic_threshold: float,
        decision_threshold_override: float | None = None,
    ) -> Dict[str, Any]:
        active_config = json.loads(self._active_mapping_path.read_text(encoding="utf-8"))
        overlay_config = merge_mapping_rules(active_config, candidates)

        aligner = FeatureOntologyAligner(singleton=False)
        aligner.load_config_from_dict(overlay_config)
        facade = DetectionFacade(
            hub=self._hub,
            aligner=aligner,
            graph_evolver=self._graph_evolver,
            evidence_builder=self._evidence_builder,
            logger=self._logger,
        )

        profile_key = str(decision_profile or "").strip().lower()
        dataset_root_str = self._dataset_profile_roots.get(profile_key) or self._dataset_profile_roots.get("default")
        if not dataset_root_str:
            raise ValueError("No dataset root configured for candidate benchmarking")
        dataset_root = Path(dataset_root_str)

        samples = collect_samples(
            dataset_root=dataset_root,
            limit_per_class=None,
            sample_per_class=sample_per_class,
            seed=42,
        )

        records: List[PredictionRecord] = []
        for sample in samples:
            response = facade.execute(
                DetectRequest(
                    image_bytes=sample.path.read_bytes(),
                    auto_evolve_enabled=False,
                    semantic_threshold=semantic_threshold,
                    use_llm_generation=False,
                    decision_profile=decision_profile,
                    decision_threshold_override=decision_threshold_override,
                )
            )
            records.append(_result_to_prediction_record(sample.path, sample.truth_label, response))

        summary = compute_summary(records)
        audit_summary = compute_audit_summary(records)
        threshold_calibration = find_recommended_threshold(records)
        return {
            "dataset_root": str(dataset_root),
            "sample_per_class": sample_per_class,
            "summary": {
                "total_samples": summary.total_samples,
                "valid_predictions": summary.valid_predictions,
                "accuracy_valid": summary.accuracy_valid,
                "balanced_accuracy": summary.balanced_accuracy,
                "f1_fake": summary.f1_fake,
                "precision_fake": summary.precision_fake,
                "recall_fake": summary.recall_fake,
                "specificity_real": summary.specificity_real,
            },
            "audit_summary": audit_summary,
            "threshold_calibration": threshold_calibration,
        }
