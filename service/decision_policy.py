from __future__ import annotations

import re
from typing import Mapping, Optional, Tuple

from detector_config import AdaptiveFusionConfig
from detectors.forensics_utils import clamp01


def normalize_profile_name(name: Optional[str]) -> str:
    if not name:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return normalized.strip("_")


def resolve_decision_threshold(
    *,
    base_threshold: float,
    profile_name: Optional[str],
    override_threshold: Optional[float],
    profile_thresholds: Mapping[str, float],
) -> Tuple[float, str]:
    if override_threshold is not None:
        return clamp01(float(override_threshold)), "override"

    profile_key = normalize_profile_name(profile_name)
    if profile_key:
        profile_threshold = profile_thresholds.get(profile_key)
        if profile_threshold is not None:
            return clamp01(float(profile_threshold)), "profile"

    return clamp01(float(base_threshold)), "default"


def compute_adaptive_fusion(
    *,
    primary_score: float,
    auxiliary_score: Optional[float],
    portrait_confidence: float,
    decision_threshold: float,
    config: AdaptiveFusionConfig,
) -> dict:
    primary = clamp01(float(primary_score))
    if auxiliary_score is None:
        margin = abs(primary - float(decision_threshold))
        return {
            "fused_score": primary,
            "blend_ratio": 0.0,
            "shift_indicator": 0.0,
            "primary_margin": round(float(margin), 6),
            "mode": "primary_only",
        }

    auxiliary = clamp01(float(auxiliary_score))
    margin = abs(primary - float(decision_threshold))
    gap = abs(primary - auxiliary)
    portrait_risk = 1.0 - clamp01(float(portrait_confidence))
    margin_reference = max(float(config.margin_reference), 1e-6)
    margin_risk = 1.0 - clamp01(margin / margin_reference)

    shift = clamp01(
        float(config.gap_weight) * gap
        + float(config.portrait_weight) * portrait_risk
        + float(config.margin_weight) * margin_risk
    )
    base_blend = clamp01(float(config.base_blend))
    max_blend = clamp01(float(config.max_blend))
    if max_blend < base_blend:
        max_blend = base_blend
    blend = clamp01(base_blend + (max_blend - base_blend) * shift)
    fused_score = clamp01((1.0 - blend) * primary + blend * auxiliary)

    return {
        "fused_score": round(float(fused_score), 6),
        "blend_ratio": round(float(blend), 6),
        "shift_indicator": round(float(shift), 6),
        "primary_margin": round(float(margin), 6),
        "primary_score": round(float(primary), 6),
        "auxiliary_score": round(float(auxiliary), 6),
        "mode": "adaptive_fusion",
    }


def compute_graph_coupling(
    *,
    primary_score: float,
    decision_threshold: float,
    graph_score: Optional[float],
    evidence_count: int,
    base_graph_weight: float,
) -> dict:
    primary = clamp01(float(primary_score))
    if graph_score is None or int(evidence_count) <= 0:
        return {
            "coupled_score": round(float(primary), 6),
            "influence_weight": 0.0,
            "alignment_score": 0.0,
            "boundary_factor": 0.0,
        }

    graph = clamp01(float(graph_score))
    threshold = clamp01(float(decision_threshold))
    agreement = 1.0 - abs(primary - graph)
    activation_strength = clamp01(float(evidence_count) / 3.0)
    alignment = clamp01(
        0.55 * agreement
        + 0.30 * graph
        + 0.15 * activation_strength
    )

    margin = abs(primary - threshold)
    boundary_factor = 1.0 - clamp01(margin / 0.20)
    influence = clamp01(
        clamp01(float(base_graph_weight))
        * (0.45 + 0.55 * alignment)
        * (0.35 + 0.65 * boundary_factor)
    )
    coupled = clamp01((1.0 - influence) * primary + influence * graph)
    return {
        "coupled_score": round(float(coupled), 6),
        "influence_weight": round(float(influence), 6),
        "alignment_score": round(float(alignment), 6),
        "boundary_factor": round(float(boundary_factor), 6),
    }
