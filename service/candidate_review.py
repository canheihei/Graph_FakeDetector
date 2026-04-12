from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


class CandidateReviewFacade:
    def __init__(
        self,
        *,
        candidate_store: CandidateStore,
        candidate_graph_store: CandidateGraphStore,
        benchmark_runner: CandidateBenchmarkRunner,
        mapping_config_path: Path | str,
    ) -> None:
        self._candidate_store = candidate_store
        self._candidate_graph_store = candidate_graph_store
        self._benchmark_runner = benchmark_runner
        self._mapping_config_path = Path(mapping_config_path)
        self._review_config = get_candidate_review_config()

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

        for candidate in selected:
            promotion = dict(candidate.get("promotion", {}) or {})
            if not bool(promotion.get("eligible", False)):
                raise WorkflowError(
                    f"Candidate '{candidate.get('candidate_id')}' has not passed benchmark gating",
                    400,
                )

        promote_candidate_rules(self._mapping_config_path, selected)
        promoted_at = datetime.now(timezone.utc).isoformat()
        promoted_ids = []
        for candidate in selected:
            updated = dict(candidate)
            updated["status"] = "promoted"
            updated["approval_state"] = "approved"
            promotion = dict(updated.get("promotion", {}) or {})
            promotion["eligible"] = True
            promotion["promoted_at"] = promoted_at
            updated["promotion"] = promotion
            self._candidate_store.replace_item(str(updated.get("candidate_id")), updated)
            promoted_ids.append(str(updated.get("candidate_id")))

        self._candidate_graph_store.update_status(promoted_ids, "promoted")
        return {
            "status": "success",
            "promoted_count": len(promoted_ids),
            "candidate_ids": promoted_ids,
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
