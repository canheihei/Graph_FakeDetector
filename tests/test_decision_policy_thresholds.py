from service.decision_policy import (
    AdaptiveFusionConfig,
    compute_adaptive_fusion,
    resolve_decision_threshold,
)


def test_resolve_threshold_prefers_override():
    threshold, source = resolve_decision_threshold(
        base_threshold=0.38,
        profile_name="celeb_df",
        override_threshold=0.57,
        profile_thresholds={"celeb_df": 0.62},
    )

    assert abs(threshold - 0.57) < 1e-6
    assert source == "override"


def test_resolve_threshold_uses_profile_when_no_override():
    threshold, source = resolve_decision_threshold(
        base_threshold=0.38,
        profile_name="dfdc",
        override_threshold=None,
        profile_thresholds={"dfdc": 0.33},
    )

    assert abs(threshold - 0.33) < 1e-6
    assert source == "profile"


def test_adaptive_blend_rises_under_domain_shift():
    config = AdaptiveFusionConfig(
        base_blend=0.08,
        max_blend=0.72,
        gap_weight=0.65,
        portrait_weight=0.20,
        margin_weight=0.15,
        margin_reference=0.12,
    )

    low_shift = compute_adaptive_fusion(
        primary_score=0.52,
        auxiliary_score=0.50,
        portrait_confidence=0.92,
        decision_threshold=0.38,
        config=config,
    )
    high_shift = compute_adaptive_fusion(
        primary_score=0.85,
        auxiliary_score=0.15,
        portrait_confidence=0.55,
        decision_threshold=0.38,
        config=config,
    )

    assert high_shift["blend_ratio"] > low_shift["blend_ratio"]
    assert high_shift["shift_indicator"] > low_shift["shift_indicator"]


def test_no_auxiliary_signal_keeps_primary_score():
    fused = compute_adaptive_fusion(
        primary_score=0.74,
        auxiliary_score=None,
        portrait_confidence=0.88,
        decision_threshold=0.38,
        config=AdaptiveFusionConfig(),
    )

    assert abs(fused["fused_score"] - 0.74) < 1e-6
    assert abs(fused["blend_ratio"] - 0.0) < 1e-6
    assert fused["mode"] == "primary_only"
