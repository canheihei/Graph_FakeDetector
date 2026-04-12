import pytest

from service.candidate_benchmark import (
    build_candidate_active_graph_payload,
    ensure_no_candidate_conflicts,
    merge_mapping_rules,
)


def test_merge_mapping_rules_replaces_rule_for_same_detector_feature():
    active = {
        "version": "1.4",
        "rules": [
            {
                "detector": "FFTDetector",
                "feature": "patch_inconsistency",
                "subdomain_label": "旧节点",
                "subdomain_id": "old-id",
            }
        ],
    }
    candidate = {
        "mapping_candidate": {
            "detector": "FFTDetector",
            "feature": "patch_inconsistency",
            "subdomain_label": "新节点",
            "subdomain_id": "new-id",
        }
    }

    merged = merge_mapping_rules(active, [candidate])

    assert merged["rules"][0]["subdomain_label"] == "新节点"
    assert merged["rules"][0]["subdomain_id"] == "new-id"
    assert merged["rules"][0]["evidence_enabled"] is True


def test_ensure_no_candidate_conflicts_rejects_multiple_candidates_for_same_feature():
    selected = [
        {
            "candidate_id": "c1",
            "mapping_candidate": {"detector": "FFTDetector", "feature": "patch_inconsistency"},
        },
        {
            "candidate_id": "c2",
            "mapping_candidate": {"detector": "FFTDetector", "feature": "patch_inconsistency"},
        },
    ]

    with pytest.raises(ValueError, match="multiple candidates"):
        ensure_no_candidate_conflicts(selected)


def test_build_candidate_active_graph_payload_uses_candidate_graph_identity():
    payload = build_candidate_active_graph_payload(
        {
            "graph_candidate": {
                "main_domain": "域泛化",
                "specific_domain": "内容异常域",
                "subdomain_name": "全局伪造概率异常",
                "canonical_name": "global_fake_probability_anomaly",
                "describe": "候选描述",
                "candidate_subdomain_id": "candidate-sub-id",
            }
        }
    )

    assert payload["specific_domain"] == "内容异常域"
    assert payload["subdomain"][0]["sub_id"] == "candidate-sub-id"
    assert payload["semantic_source"] == "candidate_benchmark_overlay"
