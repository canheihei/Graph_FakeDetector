from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import uuid

from detector_config import get_candidate_review_config
from service.candidate_benchmark import (
    CandidateBenchmarkRunner,
    ensure_no_candidate_conflicts,
    promote_candidate_rules,
)
from service.candidate_generation import (
    generate_candidate_items,
    should_generate_candidates,
)
from service.candidate_graph import CandidateGraphStore
from service.candidate_store import CandidateStore
from service.facades import WorkflowError


@dataclass(frozen=True)
class CandidateRequest:
    detect_result: Dict[str, Any]
    source_sample_name: str = ""
    decision_profile: str | None = None


@dataclass(frozen=True)
class CandidateUpdateRequest:
    candidate_id: str
    graph_candidate: Dict[str, Any] | None = None
    mapping_candidate: Dict[str, Any] | None = None
    status: str | None = None
    approval_state: str | None = None


@dataclass(frozen=True)
class CandidateBenchmarkRequest:
    candidate_ids: List[str]
    mode: str
    decision_profile: str | None = None
    sample_per_class: int | None = None
    semantic_threshold: float = 0.80
    decision_threshold_override: float | None = None


@dataclass(frozen=True)
class CandidatePromoteRequest:
    candidate_ids: List[str]


@dataclass(frozen=True)
class CandidateDeleteRequest:
    candidate_ids: List[str]


class CandidateReviewFacade:
    def __init__(
        self,
        *,
        candidate_store: CandidateStore,
        candidate_graph_store: CandidateGraphStore,
        benchmark_runner: CandidateBenchmarkRunner,
        mapping_config_path: Path | str,
        graph_writer,
        neo4j_client,
        logger=None,
        aligner=None,
    ) -> None:
        self._candidate_store = candidate_store
        self._candidate_graph_store = candidate_graph_store
        self._benchmark_runner = benchmark_runner
        self._mapping_config_path = Path(mapping_config_path)
        self._review_config = get_candidate_review_config()
        self._graph_writer = graph_writer
        self._neo4j_client = neo4j_client
        self._logger = logger
        self._aligner = aligner

    def generate(self, request: CandidateRequest) -> Dict[str, Any]:
        if not should_generate_candidates(request.detect_result):
            raise WorkflowError("Current detect result is not eligible for candidate generation", 400)

        items = generate_candidate_items(
            detect_result=request.detect_result,
            source_sample_name=request.source_sample_name,
            decision_profile=request.decision_profile,
        )
        if not items:
            raise WorkflowError("No candidate mappings were generated from this detect result", 400)

        self._candidate_store.append_items(items)
        self._candidate_graph_store.persist_candidates(items)
        return {
            "status": "success",
            "count": len(items),
            "items": items,
        }

    def list_items(self, *, status: str | None = None) -> Dict[str, Any]:
        items = self._candidate_store.list_items(status=status)
        return {
            "status": "success",
            "count": len(items),
            "items": items,
        }

    def update_item(self, request: CandidateUpdateRequest) -> Dict[str, Any]:
        current = self._candidate_store.get_item(request.candidate_id)
        if current is None:
            raise WorkflowError(f"Candidate '{request.candidate_id}' not found", 404)
        if str(current.get("status", "pending")) not in {"pending"}:
            raise WorkflowError("Only pending candidates can be edited", 400)

        updated = dict(current)
        if request.graph_candidate is not None:
            graph = dict(updated.get("graph_candidate", {}) or {})
            graph.update(request.graph_candidate)
            updated["graph_candidate"] = graph
        if request.mapping_candidate is not None:
            mapping = dict(updated.get("mapping_candidate", {}) or {})
            mapping.update(request.mapping_candidate)
            updated["mapping_candidate"] = mapping
        if request.status is not None:
            updated["status"] = request.status
        if request.approval_state is not None:
            updated["approval_state"] = request.approval_state

        graph = dict(updated.get("graph_candidate", {}) or {})
        mapping = dict(updated.get("mapping_candidate", {}) or {})
        if graph.get("subdomain_name"):
            mapping["subdomain_label"] = graph.get("subdomain_name")
        if graph.get("candidate_subdomain_id"):
            mapping["subdomain_id"] = graph.get("candidate_subdomain_id")
        updated["mapping_candidate"] = mapping

        saved = self._candidate_store.replace_item(request.candidate_id, updated)
        self._candidate_graph_store.persist_candidates([saved])
        if request.status is not None:
            self._candidate_graph_store.update_status([request.candidate_id], request.status)
        return {
            "status": "success",
            "item": saved,
        }

    def benchmark(self, request: CandidateBenchmarkRequest) -> Dict[str, Any]:
        selected = self._get_selected_candidates(request.candidate_ids)
        ensure_no_candidate_conflicts(selected)
        self._ensure_status_allowed(selected, allowed_statuses={"pending"}, action="benchmark")

        mode = str(request.mode or "").strip().lower()
        if mode not in {"quick", "formal"}:
            raise WorkflowError("benchmark mode must be 'quick' or 'formal'", 400)

        mode_config = self._review_config.quick if mode == "quick" else self._review_config.formal
        sample_per_class = request.sample_per_class or mode_config.sample_per_class
        decision_profile = request.decision_profile or self._resolve_decision_profile(selected)

        result = self._benchmark_runner.run(
            candidates=selected,
            decision_profile=decision_profile,
            sample_per_class=sample_per_class,
            semantic_threshold=request.semantic_threshold,
            decision_threshold_override=request.decision_threshold_override,
        )
        summary = result.get("summary", {})
        passed = bool(
            float(summary.get("accuracy_valid", 0.0) or 0.0) >= float(mode_config.min_accuracy_valid)
            and float(summary.get("balanced_accuracy", 0.0) or 0.0) >= float(mode_config.min_balanced_accuracy)
        )
        result["passed"] = passed
        result["decision_profile"] = decision_profile

        for candidate in selected:
            updated = dict(candidate)
            benchmarks = dict(updated.get("benchmarks", {}) or {})
            benchmarks[mode] = result
            updated["benchmarks"] = benchmarks
            updated["status"] = "benchmarked"
            promotion = dict(updated.get("promotion", {}) or {})
            promotion["eligible"] = bool(
                passed
                or (benchmarks.get("quick") or {}).get("passed")
                or (benchmarks.get("formal") or {}).get("passed")
            )
            updated["promotion"] = promotion
            self._candidate_store.replace_item(str(updated.get("candidate_id")), updated)

        return {"status": "success", "mode": mode, "result": result}

    def promote(self, request: CandidatePromoteRequest) -> Dict[str, Any]:
        selected = self._get_selected_candidates(request.candidate_ids)
        ensure_no_candidate_conflicts(selected)
        self._ensure_status_allowed(selected, allowed_statuses={"benchmarked"}, action="promote")

        for candidate in selected:
            promotion = dict(candidate.get("promotion", {}) or {})
            if not bool(promotion.get("eligible", False)):
                raise WorkflowError(
                    f"Candidate '{candidate.get('candidate_id')}' has not passed benchmark gating",
                    400,
                )

        promoted_candidates: List[Dict[str, Any]] = []
        promotion_logs: List[Dict[str, Any]] = []
        for candidate in selected:
            promoted_candidate, promote_log = self._promote_candidate_to_active_graph(candidate)
            promoted_candidates.append(promoted_candidate)
            promotion_logs.append(promote_log)

        promote_candidate_rules(self._mapping_config_path, promoted_candidates)
        if self._aligner is not None:
            self._aligner.load_config(str(self._mapping_config_path))
            if self._logger is not None:
                self._logger.info(
                    "[PROMOTE] reloaded active aligner from %s",
                    self._mapping_config_path,
                )
        promoted_at = datetime.now(timezone.utc).isoformat()
        promoted_ids = []
        for candidate, promote_log in zip(promoted_candidates, promotion_logs):
            updated = dict(candidate)
            updated["status"] = "promoted"
            updated["approval_state"] = "approved"
            promotion = dict(updated.get("promotion", {}) or {})
            promotion["eligible"] = True
            promotion["promoted_at"] = promoted_at
            promotion["active_specific_domain"] = promote_log["active_graph"]["specific_domain"]
            promotion["active_subdomain_id"] = promote_log["active_graph"]["subdomain_id"]
            promotion["active_subdomain_name"] = promote_log["active_graph"]["subdomain_name"]
            promotion["last_log"] = promote_log
            updated["promotion"] = promotion
            self._candidate_store.replace_item(str(updated.get("candidate_id")), updated)
            promoted_ids.append(str(updated.get("candidate_id")))

        self._candidate_graph_store.update_status(promoted_ids, "promoted")
        return {
            "status": "success",
            "promoted_count": len(promoted_ids),
            "candidate_ids": promoted_ids,
            "logs": promotion_logs,
        }

    def delete(self, request: CandidateDeleteRequest) -> Dict[str, Any]:
        deleted = self._candidate_store.delete_items(request.candidate_ids)
        self._candidate_graph_store.delete_candidates(request.candidate_ids)
        return {
            "status": "success",
            "deleted_count": deleted,
            "candidate_ids": list(request.candidate_ids),
        }

    def _get_selected_candidates(self, candidate_ids: Iterable[str]) -> List[Dict[str, Any]]:
        serialized_ids = [str(item) for item in candidate_ids if str(item).strip()]
        if not serialized_ids:
            raise WorkflowError("No candidate ids provided", 400)
        selected = []
        for candidate_id in serialized_ids:
            item = self._candidate_store.get_item(candidate_id)
            if item is None:
                raise WorkflowError(f"Candidate '{candidate_id}' not found", 404)
            selected.append(item)
        return selected

    @staticmethod
    def _resolve_decision_profile(selected: Iterable[Dict[str, Any]]) -> Optional[str]:
        profiles = {
            str((item.get("source", {}) or {}).get("decision_profile", "") or "").strip()
            for item in selected
        }
        profiles.discard("")
        if len(profiles) == 1:
            return next(iter(profiles))
        return None

    @staticmethod
    def _ensure_status_allowed(
        selected: Iterable[Dict[str, Any]],
        *,
        allowed_statuses: set[str],
        action: str,
    ) -> None:
        disallowed = [
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "status": str(item.get("status", "pending")),
            }
            for item in selected
            if str(item.get("status", "pending")) not in allowed_statuses
        ]
        if disallowed:
            detail = ", ".join(
                f"{item['candidate_id']}[{item['status']}]"
                for item in disallowed
            )
            raise WorkflowError(
                f"Only {', '.join(sorted(allowed_statuses))} candidates can {action}: {detail}",
                400,
            )

    def _promote_candidate_to_active_graph(self, candidate: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        payload = self._build_active_graph_payload(candidate)
        merge_stats = self._graph_writer.write(payload, semantic_threshold=0.90) or {}
        target_specific_domain = (
            str(merge_stats.get("specific_domain_name", "") or "").strip()
            or payload["specific_domain"]
        )
        resolved_subdomain = self._neo4j_client.find_subdomain_record(
            specific_domain_name=target_specific_domain,
            sub_id=str(payload["subdomain"][0]["sub_id"]),
            canonical_name=str(payload["subdomain"][0].get("canonical_name", "") or ""),
            sub_name=str(payload["subdomain"][0]["name"]),
        )
        if resolved_subdomain is None:
            raise WorkflowError(
                f"Promoted subdomain '{payload['subdomain'][0]['name']}' could not be resolved in active graph",
                500,
            )

        specific_record = next(
            (
                item for item in self._neo4j_client.list_specific_domains(include_main_domain=True)
                if str(item.get("name", "") or "").strip() == target_specific_domain
            ),
            None,
        )
        updated_candidate = dict(candidate)
        graph = dict(updated_candidate.get("graph_candidate", {}) or {})
        graph["main_domain"] = resolved_subdomain.get("main_domain") or graph.get("main_domain") or "域泛化"
        graph["specific_domain"] = target_specific_domain
        graph["subdomain_name"] = resolved_subdomain.get("display_name") or resolved_subdomain.get("name") or graph.get("subdomain_name")
        graph["canonical_name"] = resolved_subdomain.get("canonical_name") or graph.get("canonical_name")
        graph["describe"] = resolved_subdomain.get("describe") or graph.get("describe")
        graph["candidate_subdomain_id"] = resolved_subdomain["sub_id"]
        updated_candidate["graph_candidate"] = graph

        mapping = dict(updated_candidate.get("mapping_candidate", {}) or {})
        mapping_before = dict(mapping)
        mapping["subdomain_id"] = resolved_subdomain["sub_id"]
        mapping["subdomain_label"] = graph["subdomain_name"]
        updated_candidate["mapping_candidate"] = mapping

        promote_log = {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "mapping_key": f"{mapping.get('detector', '')}:{mapping.get('feature', '')}",
            "mapping_before": {
                "subdomain_id": mapping_before.get("subdomain_id"),
                "subdomain_label": mapping_before.get("subdomain_label"),
            },
            "graph_merge": {
                "specific_domain_reused": bool(merge_stats.get("specific_domain_reused", False)),
                "specific_domain_name": target_specific_domain,
                "subdomain_matched": int(merge_stats.get("subdomain_matched", 0) or 0),
                "subdomain_created": int(merge_stats.get("subdomain_created", 0) or 0),
                "semantic_threshold": float(merge_stats.get("semantic_threshold", 0.90) or 0.90),
            },
            "active_graph": {
                "main_domain": graph["main_domain"],
                "specific_domain": target_specific_domain,
                "specific_id": (specific_record or {}).get("id"),
                "subdomain_id": resolved_subdomain["sub_id"],
                "subdomain_name": graph["subdomain_name"],
                "canonical_name": graph.get("canonical_name"),
            },
            "mapping_after": {
                "subdomain_id": mapping["subdomain_id"],
                "subdomain_label": mapping["subdomain_label"],
            },
        }
        if self._logger is not None:
            self._logger.info(
                "[PROMOTE] %s active_subdomain=%s specific_domain=%s reused_specific=%s matched=%s created=%s",
                promote_log["mapping_key"],
                promote_log["active_graph"]["subdomain_id"],
                promote_log["active_graph"]["specific_domain"],
                promote_log["graph_merge"]["specific_domain_reused"],
                promote_log["graph_merge"]["subdomain_matched"],
                promote_log["graph_merge"]["subdomain_created"],
            )
        return updated_candidate, promote_log

    @staticmethod
    def _build_active_graph_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
        graph = dict(candidate.get("graph_candidate", {}) or {})
        main_domain = str(graph.get("main_domain", "") or "").strip() or "域泛化"
        specific_domain = str(graph.get("specific_domain", "") or "").strip() or "未分类候选域"
        specific_id = str(graph.get("specific_id", "") or "").strip() or str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"candidate-promote::{specific_domain}")
        )
        sub_name = str(graph.get("subdomain_name", "") or "").strip() or "未命名候选节点"
        describe = str(graph.get("describe", "") or "").strip() or (
            f"该节点表示“{specific_domain}”下经审批晋级的稳定伪造语义证据，名称为“{sub_name}”。"
        )
        sub_id = str(graph.get("candidate_subdomain_id", "") or "").strip() or str(uuid.uuid4())
        return {
            "main_domain": main_domain,
            "main_describe": f"候选晋级合并到正式图谱的主域“{main_domain}”。",
            "specific_domain": specific_domain,
            "describe": f"候选晋级合并到正式图谱的语义域“{specific_domain}”。",
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
            "semantic_source": "candidate_promote",
            "semantic_prompt": "",
            "semantic_version": "graph_semantics_v2_candidate_promote",
        }
