from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import uuid

from alignment.aligner import FeatureOntologyAligner
from scripts.benchmark.visualize_detect_benchmark import (
    PredictionRecord,
    collect_samples,
    compute_audit_summary,
    compute_summary,
    find_recommended_threshold,
)
from service.graph_semantics import match_existing_subdomain
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


def _normalize_candidate_rule(candidate: Dict[str, Any], *, activate_for_use: bool = True) -> Dict[str, Any]:
    mapping = dict(candidate.get("mapping_candidate", {}) or {})
    graph = dict(candidate.get("graph_candidate", {}) or {})
    mapping.setdefault("subdomain_id", graph.get("candidate_subdomain_id") or graph.get("subdomain_id"))
    mapping.setdefault("subdomain_label", graph.get("subdomain_name"))
    mapping["evidence_enabled"] = bool(mapping.get("evidence_enabled", False) or activate_for_use)
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
        normalized = _normalize_candidate_rule(candidate, activate_for_use=True)
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


def build_candidate_active_graph_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    graph = dict(candidate.get("graph_candidate", {}) or {})
    main_domain = str(graph.get("main_domain", "") or "").strip() or "域泛化"
    specific_domain = str(graph.get("specific_domain", "") or "").strip() or "未分类候选域"
    specific_id = str(graph.get("specific_id", "") or "").strip() or str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"candidate-benchmark::{specific_domain}")
    )
    sub_name = str(graph.get("subdomain_name", "") or "").strip() or "未命名候选节点"
    describe = str(graph.get("describe", "") or "").strip() or (
        f"该节点表示“{specific_domain}”下临时评测合并的候选伪造语义证据，名称为“{sub_name}”。"
    )
    sub_id = str(graph.get("candidate_subdomain_id", "") or "").strip() or str(uuid.uuid4())
    return {
        "main_domain": main_domain,
        "main_describe": f"候选 benchmark 临时合并到 active graph 的主域“{main_domain}”。",
        "specific_domain": specific_domain,
        "describe": f"候选 benchmark 临时合并到 active graph 的语义域“{specific_domain}”。",
        "specific_id": specific_id,
        "subdomain": [
            {
                "name": sub_name,
                "display_name": sub_name,
                "canonical_name": str(graph.get("canonical_name", "") or "").strip() or None,
                "describe": describe,
                "sub_id": sub_id,
            }
        ],
        "semantic_source": "candidate_benchmark_overlay",
        "semantic_prompt": "",
        "semantic_version": "graph_semantics_v2_candidate_benchmark",
    }


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
        graph_writer,
        neo4j_client,
    ) -> None:
        self._hub = hub
        self._graph_evolver = graph_evolver
        self._evidence_builder = evidence_builder
        self._logger = logger
        self._dataset_profile_roots = dict(dataset_profile_roots)
        self._active_mapping_path = Path(active_mapping_path)
        self._graph_writer = graph_writer
        self._neo4j_client = neo4j_client

    def run(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        decision_profile: str | None,
        sample_per_class: int,
        semantic_threshold: float,
        decision_threshold_override: float | None = None,
    ) -> Dict[str, Any]:
        cleanup_token = f"benchmark-overlay-{uuid.uuid4()}"
        overlay_candidates = self._merge_candidates_into_temporary_active_graph(
            candidates,
            cleanup_token=cleanup_token,
        )
        try:
            active_config = json.loads(self._active_mapping_path.read_text(encoding="utf-8"))
            overlay_config = merge_mapping_rules(active_config, overlay_candidates)

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
                "overlay_logs": [
                    candidate.get("benchmark_overlay", {})
                    for candidate in overlay_candidates
                ],
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
        finally:
            self._cleanup_temporary_overlay_graph(cleanup_token)

    def _merge_candidates_into_temporary_active_graph(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        cleanup_token: str,
    ) -> List[Dict[str, Any]]:
        merged_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            graph = dict(candidate.get("graph_candidate", {}) or {})
            target_specific_domain = str(graph.get("specific_domain", "") or "").strip() or "未分类候选域"
            existing_subdomains = self._neo4j_client.list_subdomain_records(target_specific_domain)
            sub_payload = {
                "name": str(graph.get("subdomain_name", "") or "").strip() or "未命名候选节点",
                "display_name": str(graph.get("subdomain_name", "") or "").strip() or "未命名候选节点",
                "canonical_name": str(graph.get("canonical_name", "") or "").strip() or None,
                "describe": str(graph.get("describe", "") or "").strip(),
                "sub_id": str(graph.get("candidate_subdomain_id", "") or "").strip() or str(uuid.uuid4()),
            }
            matched = match_existing_subdomain(
                sub_payload,
                existing_subdomains,
                semantic_threshold=0.90,
            )
            specific_exists = any(
                str(item.get("name", "") or "").strip() == target_specific_domain
                for item in self._neo4j_client.list_specific_domains(include_main_domain=True)
            )
            specific_created = False
            subdomain_created = False
            if matched is None:
                self._neo4j_client.execute_write(
                    self._create_temporary_overlay_subgraph,
                    cleanup_token=cleanup_token,
                    payload=build_candidate_active_graph_payload(candidate),
                )
                specific_created = not specific_exists
                subdomain_created = True
                resolved_subdomain = self._neo4j_client.find_subdomain_record(
                    specific_domain_name=target_specific_domain,
                    sub_id=sub_payload["sub_id"],
                    canonical_name=sub_payload.get("canonical_name") or "",
                    sub_name=sub_payload["name"],
                )
            else:
                resolved_subdomain = matched

            if resolved_subdomain is None:
                raise ValueError(
                    f"Benchmark overlay subdomain '{sub_payload['name']}' could not be resolved"
                )
            updated_candidate = dict(candidate)
            graph["main_domain"] = resolved_subdomain.get("main_domain") or graph.get("main_domain") or "域泛化"
            graph["specific_domain"] = target_specific_domain
            graph["subdomain_name"] = resolved_subdomain.get("display_name") or resolved_subdomain.get("name") or graph.get("subdomain_name")
            graph["canonical_name"] = resolved_subdomain.get("canonical_name") or graph.get("canonical_name")
            graph["describe"] = resolved_subdomain.get("describe") or graph.get("describe")
            graph["candidate_subdomain_id"] = resolved_subdomain["sub_id"]
            updated_candidate["graph_candidate"] = graph

            mapping = dict(updated_candidate.get("mapping_candidate", {}) or {})
            mapping["subdomain_id"] = resolved_subdomain["sub_id"]
            mapping["subdomain_label"] = graph["subdomain_name"]
            updated_candidate["mapping_candidate"] = mapping
            updated_candidate["benchmark_overlay"] = {
                "cleanup_token": cleanup_token,
                "specific_domain_reused": not specific_created,
                "specific_domain_name": target_specific_domain,
                "subdomain_matched": 0 if subdomain_created else 1,
                "subdomain_created": 1 if subdomain_created else 0,
                "active_subdomain_id": resolved_subdomain["sub_id"],
                "active_subdomain_name": graph["subdomain_name"],
                "temporary_overlay": True,
            }
            merged_candidates.append(updated_candidate)
        return merged_candidates

    @staticmethod
    def _create_temporary_overlay_subgraph(tx, *, cleanup_token: str, payload: Dict[str, Any]) -> None:
        main_domain = str(payload.get("main_domain", "") or "").strip() or "域泛化"
        specific_domain = str(payload.get("specific_domain", "") or "").strip() or "未分类候选域"
        specific_id = str(payload.get("specific_id", "") or "").strip() or str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"candidate-benchmark::{specific_domain}")
        )
        sub = dict((payload.get("subdomain") or [{}])[0])
        tx.run(
            """
            MERGE (m:MainDomain {name: $main_domain})
            ON CREATE SET
                m.describe = $main_describe,
                m.benchmark_overlay_token = $cleanup_token
            WITH m
            MERGE (s:SpecificDomain {name: $specific_domain})
            ON CREATE SET
                s.specific_id = $specific_id,
                s.describe = $specific_describe,
                s.benchmark_overlay_token = $cleanup_token
            MERGE (s)-[:KINDS_OF]->(m)
            MERGE (sub:SubDomain {sub_id: $sub_id})
            ON CREATE SET
                sub.name = $sub_name,
                sub.display_name = $sub_display_name,
                sub.canonical_name = $sub_canonical_name,
                sub.describe = $sub_describe,
                sub.semantic_source = 'candidate_benchmark_overlay',
                sub.semantic_version = 'graph_semantics_v2_candidate_benchmark',
                sub.benchmark_overlay_token = $cleanup_token
            MERGE (sub)-[:SPECIFIC_OF]->(s)
            """,
            cleanup_token=cleanup_token,
            main_domain=main_domain,
            main_describe=str(payload.get("main_describe", "") or ""),
            specific_domain=specific_domain,
            specific_id=specific_id,
            specific_describe=str(payload.get("describe", "") or ""),
            sub_id=str(sub.get("sub_id", "")),
            sub_name=str(sub.get("name", "")),
            sub_display_name=str(sub.get("display_name", sub.get("name", ""))),
            sub_canonical_name=str(sub.get("canonical_name", "") or ""),
            sub_describe=str(sub.get("describe", "") or ""),
        )

    def _cleanup_temporary_overlay_graph(self, cleanup_token: str) -> None:
        self._neo4j_client.execute_write(
            self._cleanup_temporary_overlay_graph_tx,
            cleanup_token=cleanup_token,
        )

    @staticmethod
    def _cleanup_temporary_overlay_graph_tx(tx, *, cleanup_token: str) -> None:
        tx.run(
            """
            MATCH (sub:SubDomain {benchmark_overlay_token: $cleanup_token})
            DETACH DELETE sub
            """,
            cleanup_token=cleanup_token,
        )
        tx.run(
            """
            MATCH (s:SpecificDomain {benchmark_overlay_token: $cleanup_token})
            WHERE NOT EXISTS { MATCH (:SubDomain)-[:SPECIFIC_OF]->(s) }
            DETACH DELETE s
            """,
            cleanup_token=cleanup_token,
        )
        tx.run(
            """
            MATCH (m:MainDomain {benchmark_overlay_token: $cleanup_token})
            WHERE NOT EXISTS { MATCH (:SpecificDomain)-[:KINDS_OF]->(m) }
            DETACH DELETE m
            """,
            cleanup_token=cleanup_token,
        )
