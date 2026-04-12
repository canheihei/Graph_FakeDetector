from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Callable, Dict, Iterable, List

from service.llm_chain import call_detect_candidate_llm, extract_candidate_feature_groups


DEFAULT_PROMPT_VERSION = "detect_candidate_mapping_v1"
FOCUS_EXCLUDED_STATUSES = {"activated"}


def should_generate_candidates(detect_result: Dict[str, Any]) -> bool:
    if str(detect_result.get("label", "")).strip().upper() != "FAKE":
        return False

    reasoning_type = str(detect_result.get("reasoning_type", "")).strip()
    if reasoning_type == "anomaly_model_only":
        return True

    evidence = detect_result.get("evidence", []) or []
    if len(evidence) == 0:
        return True

    evidence_diagnostics = detect_result.get("evidence_diagnostics", {}) or {}
    return int(evidence_diagnostics.get("unresolved_subdomains", 0) or 0) > 0


def build_focus_features(detect_result: Dict[str, Any], *, limit: int = 3) -> List[Dict[str, Any]]:
    context = detect_result.get("candidate_context", {}) or {}
    diagnostics = list(context.get("feature_diagnostics", []) or [])
    filtered = [
        item
        for item in diagnostics
        if str(item.get("status", "")).strip() not in FOCUS_EXCLUDED_STATUSES
    ]
    filtered.sort(
        key=lambda item: (
            float(item.get("priority_score", 0.0) or 0.0),
            float(item.get("raw_value", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return filtered[:limit]


def _stable_group_id(sample_name: str, detector: str, feature: str) -> str:
    raw = f"{sample_name}|{detector}|{feature}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _snake_case(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or fallback


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _allowed_features_by_detector(detect_result: Dict[str, Any]) -> Dict[str, set[str]]:
    context = detect_result.get("candidate_context", {}) or {}
    diagnostics = list(context.get("feature_diagnostics", []) or [])
    allowed: Dict[str, set[str]] = {}
    for item in diagnostics:
        detector = str(item.get("detector", "") or "").strip()
        feature = str(item.get("feature", "") or "").strip()
        if not detector or not feature:
            continue
        allowed.setdefault(detector, set()).add(feature)
    return allowed


def sanitize_candidate_alternative(
    alternative: Dict[str, Any],
    *,
    allowed_features_by_detector: Dict[str, set[str]],
) -> Dict[str, Any]:
    sanitized = dict(alternative)
    sanitized["main_domain"] = "域泛化"
    sanitized["specific_domain"] = _clip_text(
        sanitized.get("specific_domain") or "未分类候选域",
        32,
    )
    sanitized["subdomain_name"] = _clip_text(
        sanitized.get("subdomain_name") or "候选伪造证据",
        32,
    )
    sanitized["canonical_name"] = _snake_case(
        sanitized.get("canonical_name") or sanitized["subdomain_name"],
        "candidate_graph_signal",
    )
    sanitized["describe"] = _clip_text(sanitized.get("describe"), 80)
    sanitized["feature_rationale"] = _clip_text(sanitized.get("feature_rationale"), 48)
    sanitized["mapping_rationale"] = _clip_text(sanitized.get("mapping_rationale"), 48)

    context_detector = str(sanitized.get("context_detector", "") or "").strip()
    context_feature = str(sanitized.get("context_feature", "") or "").strip()
    if context_detector:
        allowed_features = allowed_features_by_detector.get(context_detector)
        if not allowed_features or context_feature not in allowed_features:
            context_detector = ""
            context_feature = ""
            sanitized["context_min_value"] = 0.0

    sanitized["context_detector"] = context_detector
    sanitized["context_feature"] = context_feature
    return sanitized


def build_candidate_items_from_llm_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("feature_groups"), list):
        return payload
    if isinstance(payload, str):
        return extract_candidate_feature_groups(payload)
    return {"feature_groups": []}


def _fallback_alternatives(focus_feature: Dict[str, Any]) -> List[Dict[str, Any]]:
    detector = str(focus_feature.get("detector", "") or "").replace("Detector", "")
    feature = str(focus_feature.get("feature", "") or "")
    feature_label = feature.replace("_", " ").strip() or "候选特征"
    base_slug = feature.lower().replace(" ", "_")
    primary_domain = "后处理痕迹域"
    if "symmetry" in base_slug or "pose" in base_slug or "lighting" in base_slug:
        primary_domain = "外观扰动域"
    elif "probability" in base_slug or "agreement" in base_slug:
        primary_domain = "内容异常域"

    return [
        {
            "main_domain": "域泛化",
            "specific_domain": primary_domain,
            "subdomain_name": f"{detector} {feature_label}异常".strip(),
            "canonical_name": f"{detector.lower()}_{base_slug}_candidate",
            "describe": f"由 {detector} 的特征 {feature} 指向的候选伪造证据节点。",
            "weight": max(float(focus_feature.get("weight", 0.70) or 0.70), 0.70),
            "activation_threshold": float(focus_feature.get("activation_threshold", 0.58) or 0.58),
            "context_detector": str(focus_feature.get("context_detector", "") or ""),
            "context_feature": str(focus_feature.get("context_feature", "") or ""),
            "context_min_value": float(focus_feature.get("context_min_value", 0.0) or 0.0),
            "feature_rationale": "Fallback candidate generated without LLM.",
            "mapping_rationale": "Keeps the current activation shape and only swaps the ontology target.",
            "prompt_version": f"{DEFAULT_PROMPT_VERSION}_fallback",
        },
        {
            "main_domain": "域泛化",
            "specific_domain": primary_domain,
            "subdomain_name": f"{feature_label}重构异常",
            "canonical_name": f"{base_slug}_reconstruction_candidate",
            "describe": f"针对特征 {feature} 的重构异常候选节点，用于补充当前证据链薄弱样本。",
            "weight": max(float(focus_feature.get("weight", 0.70) or 0.70) - 0.05, 0.55),
            "activation_threshold": max(
                float(focus_feature.get("activation_threshold", 0.58) or 0.58) - 0.05,
                0.45,
            ),
            "context_detector": str(focus_feature.get("context_detector", "") or ""),
            "context_feature": str(focus_feature.get("context_feature", "") or ""),
            "context_min_value": float(focus_feature.get("context_min_value", 0.0) or 0.0),
            "feature_rationale": "Fallback alternative with a slightly looser threshold.",
            "mapping_rationale": "Useful when the current rule is repeatedly blocked by threshold.",
            "prompt_version": f"{DEFAULT_PROMPT_VERSION}_fallback",
        },
    ]


def _build_candidate_item(
    *,
    alternative: Dict[str, Any],
    focus_feature: Dict[str, Any],
    source: Dict[str, Any],
    sample_name: str,
    rank: int,
) -> Dict[str, Any]:
    graph_candidate = {
        "main_domain": str(alternative.get("main_domain", "域泛化") or "域泛化"),
        "specific_domain": str(alternative.get("specific_domain", "未分类候选域") or "未分类候选域"),
        "subdomain_name": str(alternative.get("subdomain_name", "") or "").strip(),
        "canonical_name": str(alternative.get("canonical_name", "") or "").strip(),
        "describe": str(alternative.get("describe", "") or "").strip(),
        "candidate_subdomain_id": str(alternative.get("candidate_subdomain_id", uuid.uuid4())),
    }
    mapping_candidate = {
        "detector": str(focus_feature.get("detector", "") or ""),
        "feature": str(focus_feature.get("feature", "") or ""),
        "subdomain_id": graph_candidate["candidate_subdomain_id"],
        "subdomain_label": graph_candidate["subdomain_name"],
        "sigmoid_k": float(alternative.get("sigmoid_k", focus_feature.get("sigmoid_k", 8.0)) or 8.0),
        "sigmoid_x0": float(alternative.get("sigmoid_x0", focus_feature.get("sigmoid_x0", 0.5)) or 0.5),
        "weight": float(alternative.get("weight", focus_feature.get("weight", 0.70)) or 0.70),
        "activation_threshold": float(
            alternative.get("activation_threshold", focus_feature.get("activation_threshold", 0.58)) or 0.58
        ),
        "context_detector": str(
            alternative.get("context_detector", focus_feature.get("context_detector", "")) or ""
        ),
        "context_feature": str(
            alternative.get("context_feature", focus_feature.get("context_feature", "")) or ""
        ),
        "context_min_value": float(
            alternative.get("context_min_value", focus_feature.get("context_min_value", 0.0)) or 0.0
        ),
        "evidence_enabled": False,
    }
    return {
        "candidate_id": str(uuid.uuid4()),
        "candidate_group_id": _stable_group_id(
            sample_name,
            mapping_candidate["detector"],
            mapping_candidate["feature"],
        ),
        "status": "pending",
        "approval_state": "draft",
        "source": source,
        "graph_candidate": graph_candidate,
        "mapping_candidate": mapping_candidate,
        "llm": {
            "prompt_version": str(alternative.get("prompt_version", DEFAULT_PROMPT_VERSION)),
            "feature_rationale": str(alternative.get("feature_rationale", "") or ""),
            "mapping_rationale": str(alternative.get("mapping_rationale", "") or ""),
            "rank": rank,
        },
        "existing_rule_snapshot": {
            "status": str(focus_feature.get("status", "") or ""),
            "current_subdomain_id": focus_feature.get("current_subdomain_id"),
            "current_subdomain_label": focus_feature.get("current_subdomain_label"),
            "activation_threshold": focus_feature.get("activation_threshold"),
            "weight": focus_feature.get("weight"),
            "context_detector": focus_feature.get("context_detector"),
            "context_feature": focus_feature.get("context_feature"),
            "context_min_value": focus_feature.get("context_min_value"),
        },
        "benchmarks": {"quick": None, "formal": None},
        "promotion": {"eligible": False, "promoted_at": None},
    }


def _build_llm_payload(
    *,
    detect_result: Dict[str, Any],
    focus_features: Iterable[Dict[str, Any]],
    source_sample_name: str,
) -> Dict[str, Any]:
    focus_list = list(focus_features)
    allowed = {
        detector: sorted(features)
        for detector, features in _allowed_features_by_detector(detect_result).items()
    }
    compact_focus = [
        {
            "detector": item.get("detector"),
            "feature": item.get("feature"),
            "raw_value": item.get("raw_value"),
            "status": item.get("status"),
            "priority_score": item.get("priority_score"),
            "current_subdomain_label": item.get("current_subdomain_label"),
            "mapped_confidence": item.get("mapped_confidence"),
            "activation_threshold": item.get("activation_threshold"),
            "weight": item.get("weight"),
            "context_detector": item.get("context_detector"),
            "context_feature": item.get("context_feature"),
            "context_min_value": item.get("context_min_value"),
        }
        for item in focus_list
    ]
    return {
        "sample_name": source_sample_name,
        "decision": {
            "label": detect_result.get("label"),
            "reasoning_type": detect_result.get("reasoning_type"),
            "decision_fake_score": detect_result.get("decision_fake_score"),
            "decision_threshold": detect_result.get("decision_threshold"),
            "risk_level": detect_result.get("risk_level"),
        },
        "evidence": [
            {
                "subdomain_name": item.get("subdomain_name"),
                "specific_domain": (item.get("specific_domain", {}) or {}).get("name"),
                "score": item.get("score"),
                "confidence": item.get("confidence"),
            }
            for item in (detect_result.get("evidence", []) or [])
        ],
        "evidence_diagnostics": detect_result.get("evidence_diagnostics", {}),
        "graph_gate_diagnostics": detect_result.get("graph_gate_diagnostics", {}),
        "allowed_context_features_by_detector": allowed,
        "focus_features": compact_focus,
        "output_constraints": {
            "alternatives_per_feature": "2 preferred, 3 only if genuinely distinct",
            "max_describe_chars": 80,
            "max_rationale_chars": 48,
            "context_rule": "context_detector/context_feature must come from allowed_context_features_by_detector or be empty",
        },
    }


def _normalize_llm_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    return build_candidate_items_from_llm_payload(payload)


def generate_candidate_items(
    *,
    detect_result: Dict[str, Any],
    source_sample_name: str,
    decision_profile: str | None = None,
    llm_caller: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    focus_features = build_focus_features(detect_result)
    if not focus_features:
        return []
    allowed_features = _allowed_features_by_detector(detect_result)

    source = {
        "source_type": "detect_candidate",
        "decision_profile": decision_profile or "",
        "reasoning_type": str(detect_result.get("reasoning_type", "") or ""),
        "sample_name": source_sample_name,
        "decision_fake_score": float(detect_result.get("decision_fake_score", 0.0) or 0.0),
    }

    payload = _build_llm_payload(
        detect_result=detect_result,
        focus_features=focus_features,
        source_sample_name=source_sample_name,
    )

    groups: Dict[str, Any] = {"feature_groups": []}
    llm = llm_caller or call_detect_candidate_llm
    try:
        groups = _normalize_llm_response(llm(payload))
    except Exception:
        groups = {"feature_groups": []}

    feature_group_lookup = {
        (str(item.get("detector", "") or ""), str(item.get("feature", "") or "")): item
        for item in groups.get("feature_groups", [])
        if isinstance(item, dict)
    }

    items: List[Dict[str, Any]] = []
    for focus_feature in focus_features:
        key = (
            str(focus_feature.get("detector", "") or ""),
            str(focus_feature.get("feature", "") or ""),
        )
        group = feature_group_lookup.get(key, {})
        alternatives = list(group.get("alternatives", []) or [])[:3]
        if not alternatives:
            alternatives = _fallback_alternatives(focus_feature)
        else:
            alternatives = [
                sanitize_candidate_alternative(
                    alternative,
                    allowed_features_by_detector=allowed_features,
                )
                for alternative in alternatives
            ]
        for index, alternative in enumerate(alternatives, start=1):
            items.append(
                _build_candidate_item(
                    alternative=alternative,
                    focus_feature=focus_feature,
                    source=source,
                    sample_name=source_sample_name,
                    rank=index,
                )
            )
    return items
