from service.graph_semantics import GraphSemanticGovernance


class _FakeNeo4jClient:
    def list_specific_domains(self, include_main_domain: bool = False):
        return []

    def get_main_domain_name_by_specific_domain(self, specific_domain_name: str):
        return None

    def get_main_domain_describe(self, main_domain_name: str):
        if main_domain_name == "域泛化":
            return "唯一主域"
        return ""

    def list_main_domains(self):
        return [{"name": "域泛化", "describe": "唯一主域"}]


def test_resolve_specific_domain_reuses_single_existing_main_domain():
    governor = GraphSemanticGovernance(_FakeNeo4jClient())

    resolved = governor.resolve_specific_domain(
        candidate_name="一致性分析",
        fallback_domain={"name": "一致性分析"},
    )

    assert resolved is not None
    assert resolved.main_domain == "域泛化"
    assert resolved.main_describe == "唯一主域"
