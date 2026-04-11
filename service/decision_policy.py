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
