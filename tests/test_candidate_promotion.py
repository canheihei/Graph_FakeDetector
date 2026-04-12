import json
from pathlib import Path

from service.candidate_benchmark import promote_candidate_rules
from service.candidate_graph import CandidateGraphStore
from service.candidate_review import CandidatePromoteRequest, CandidateReviewFacade
from service.candidate_store import CandidateStore


class _FakeBenchmarkRunner:
    pass


class _FakeGraphWriter:
    def __init__(self):
        self.calls = []

    def write(self, payload, semantic_threshold=None):
        self.calls.append((payload, semantic_threshold))
        return {
            "specific_domain_reused": True,
            "specific_domain_name": payload["specific_domain"],
            "subdomain_matched": 1,
            "subdomain_created": 0,
            "total_subdomain": len(payload["subdomain"]),
            "semantic_threshold": semantic_threshold,
        }


class _FakeNeo4jClient:
    def __init__(self):
        self.updated_status = []

    def execute_write(self, fn, **kwargs):
        self.updated_status.append(kwargs)
        return None

    def list_specific_domains(self, include_main_domain=False):
        return [
            {
                "id": "specific-1",
                "name": "内容异常域",
                "describe": "正式图谱中的内容异常域",
                "main_domain": "域泛化",
                "main_describe": "正式主域",
            }
        ]

    def find_subdomain_record(
        self,
        *,
        specific_domain_name,
        sub_id="",
        canonical_name="",
        sub_name="",
    ):
        return {
            "sub_id": "active-sub-id",
            "name": "全局伪造概率异常",
            "display_name": "全局伪造概率异常",
            "canonical_name": canonical_name or "global_fake_probability_anomaly",
            "describe": "正式图谱中的已合并节点",
            "specific_domain": specific_domain_name,
            "main_domain": "域泛化",
        }


def test_promote_candidate_rules_rewrites_mapping_file_for_selected_candidate(tmp_path: Path):
    mapping_path = tmp_path / "mapping_config.json"
    mapping_path.write_text(
        json.dumps(
            {
                "version": "1.4",
                "rules": [
                    {
                        "detector": "FFTDetector",
                        "feature": "patch_inconsistency",
                        "subdomain_id": "old-id",
                        "subdomain_label": "旧节点",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    selected = [
        {
            "candidate_id": "c1",
            "mapping_candidate": {
                "detector": "FFTDetector",
                "feature": "patch_inconsistency",
                "subdomain_id": "new-id",
                "subdomain_label": "新节点",
            },
        }
    ]

    promote_candidate_rules(mapping_path, selected)
    updated = json.loads(mapping_path.read_text(encoding="utf-8"))

    assert updated["rules"][0]["subdomain_id"] == "new-id"
    assert updated["rules"][0]["subdomain_label"] == "新节点"


def test_candidate_review_promote_merges_candidate_graph_and_rewrites_rule_to_active_subdomain(tmp_path: Path):
    mapping_path = tmp_path / "mapping_config.json"
    mapping_path.write_text(
        json.dumps(
            {
                "version": "1.4",
                "rules": [
                    {
                        "detector": "CalibratedVision",
                        "feature": "fake_probability",
                        "subdomain_id": "old-id",
                        "subdomain_label": "旧节点",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = CandidateStore(tmp_path / "mapping_candidates.json")
    store.append_items(
        [
            {
                "candidate_id": "c1",
                "status": "approved",
                "approval_state": "approved",
                "graph_candidate": {
                    "main_domain": "域泛化",
                    "specific_domain": "内容异常域",
                    "subdomain_name": "全局伪造概率异常",
                    "canonical_name": "global_fake_probability_anomaly",
                    "describe": "候选描述",
                    "candidate_subdomain_id": "candidate-sub-id",
                },
                "mapping_candidate": {
                    "detector": "CalibratedVision",
                    "feature": "fake_probability",
                    "subdomain_id": "candidate-sub-id",
                    "subdomain_label": "全局伪造概率异常",
                },
                "promotion": {"eligible": True, "promoted_at": None},
            }
        ]
    )

    graph_writer = _FakeGraphWriter()
    neo4j_client = _FakeNeo4jClient()
    facade = CandidateReviewFacade(
        candidate_store=store,
        candidate_graph_store=CandidateGraphStore(neo4j_client),
        benchmark_runner=_FakeBenchmarkRunner(),
        mapping_config_path=mapping_path,
        graph_writer=graph_writer,
        neo4j_client=neo4j_client,
        logger=None,
    )

    result = facade.promote(CandidatePromoteRequest(candidate_ids=["c1"]))
    updated = json.loads(mapping_path.read_text(encoding="utf-8"))
    promoted = store.get_item("c1")

    assert updated["rules"][0]["subdomain_id"] == "active-sub-id"
    assert updated["rules"][0]["subdomain_label"] == "全局伪造概率异常"
    assert len(graph_writer.calls) == 1
    assert result["logs"][0]["active_graph"]["subdomain_id"] == "active-sub-id"
    assert result["logs"][0]["mapping_after"]["subdomain_id"] == "active-sub-id"
    assert promoted["promotion"]["active_subdomain_id"] == "active-sub-id"
