from service.candidate_generation import (
    _build_candidate_item,
    build_candidate_items_from_llm_payload,
    build_focus_features,
    sanitize_candidate_alternative,
    should_generate_candidates,
)


def test_should_generate_candidates_for_fake_model_only_detection():
    detect_result = {
        "label": "FAKE",
        "reasoning_type": "anomaly_model_only",
        "evidence": [],
        "evidence_diagnostics": {"unresolved_subdomains": 0},
    }

    assert should_generate_candidates(detect_result) is True


def test_build_focus_features_prefers_unresolved_or_blocked_features():
    detect_result = {
        "candidate_context": {
            "feature_diagnostics": [
                {
                    "detector": "FFTDetector",
                    "feature": "patch_inconsistency",
                    "raw_value": 0.71,
                    "status": "blocked_by_threshold",
                    "priority_score": 0.81,
                },
                {
                    "detector": "AppearanceDetector",
                    "feature": "symmetry_break",
                    "raw_value": 0.52,
                    "status": "activated",
                    "priority_score": 0.40,
                },
                {
                    "detector": "MetaEnsemble",
                    "feature": "max_anomaly_score",
                    "raw_value": 0.64,
                    "status": "rule_disabled",
                    "priority_score": 0.75,
                },
            ]
        }
    }

    focus = build_focus_features(detect_result)

    assert [item["feature"] for item in focus] == [
        "patch_inconsistency",
        "max_anomaly_score",
    ]


def test_build_candidate_items_salvages_complete_groups_from_truncated_json():
    truncated = """
    {
      "feature_groups": [
        {
          "detector": "CalibratedVision",
          "feature": "fake_probability",
          "alternatives": [
            {
              "main_domain": "域泛化",
              "specific_domain": "内容异常域",
              "subdomain_name": "全局伪造概率异常",
              "canonical_name": "global_fake_probability_anomaly",
              "describe": "主干模型在全局视觉表征上识别到稳定伪造偏差。",
              "weight": 0.9,
              "activation_threshold": 0.58,
              "context_detector": "",
              "context_feature": "",
              "context_min_value": 0.0,
              "sigmoid_k": 8.0,
              "sigmoid_x0": 0.5,
              "feature_rationale": "高分假样本需要稳定图谱锚点。",
              "mapping_rationale": "用于补足当前证据链为空的样本。",
              "prompt_version": "detect_candidate_mapping_v1"
            }
          ]
        },
        {
          "detector": "MetaEnsemble",
          "feature": "max_anomaly_score",
          "alternatives": [
            {
              "main_domain": "域泛化",
              "specific_domain": "后处理痕迹域"
    """

    groups = build_candidate_items_from_llm_payload(truncated)

    assert len(groups["feature_groups"]) == 1
    assert groups["feature_groups"][0]["detector"] == "CalibratedVision"


def test_sanitize_candidate_alternative_clears_unknown_context_detector():
    alternative = sanitize_candidate_alternative(
        {
            "main_domain": "risk_consolidation",
            "specific_domain": "risk_scoring",
            "subdomain_name": "综合风险评分",
            "canonical_name": "composite_risk_scoring",
            "describe": "汇聚多源特征。",
            "weight": 0.8,
            "activation_threshold": 0.55,
            "context_detector": "GraphSynthesis",
            "context_feature": "graph_consistency_score",
            "context_min_value": 0.3,
        },
        allowed_features_by_detector={
            "CalibratedVision": {"fake_probability"},
            "MetaEnsemble": {"max_anomaly_score"},
        },
    )

    assert alternative["main_domain"] == "域泛化"
    assert alternative["context_detector"] == ""
    assert alternative["context_feature"] == ""
    assert alternative["context_min_value"] == 0.0


def test_build_candidate_item_includes_created_at_timestamp():
    item = _build_candidate_item(
        focus_feature={
            "detector": "FFTDetector",
            "feature": "patch_inconsistency",
            "status": "blocked_by_threshold",
        },
        alternative={
            "main_domain": "域泛化",
            "specific_domain": "后处理痕迹域",
            "subdomain_name": "边界融合不连续",
            "canonical_name": "boundary_blending_discontinuity",
            "describe": "边界区域存在不连续融合痕迹。",
        },
        source={"sample_name": "sample_a.png"},
        sample_name="sample_a.png",
        rank=1,
    )

    assert isinstance(item["created_at"], str)
    assert "T" in item["created_at"]
    assert item["created_at"].endswith("+00:00")
