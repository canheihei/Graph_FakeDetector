from service.decision_policy import compute_graph_coupling


def test_graph_coupling_returns_zero_influence_without_evidence():
    result = compute_graph_coupling(
        primary_score=0.62,
        decision_threshold=0.60,
        graph_score=0.90,
        evidence_count=0,
        base_graph_weight=0.22,
    )

    assert result["influence_weight"] == 0.0
    assert result["coupled_score"] == 0.62
    assert result["alignment_score"] == 0.0


def test_graph_coupling_shifts_score_towards_graph_near_boundary():
    result = compute_graph_coupling(
        primary_score=0.62,
        decision_threshold=0.61,
        graph_score=0.90,
        evidence_count=3,
        base_graph_weight=0.22,
    )

    assert result["influence_weight"] > 0.0
    assert result["coupled_score"] > 0.62
    assert result["coupled_score"] <= 0.90


def test_graph_coupling_alignment_prefers_consistent_signals():
    close = compute_graph_coupling(
        primary_score=0.62,
        decision_threshold=0.61,
        graph_score=0.63,
        evidence_count=2,
        base_graph_weight=0.22,
    )
    far = compute_graph_coupling(
        primary_score=0.62,
        decision_threshold=0.61,
        graph_score=0.15,
        evidence_count=2,
        base_graph_weight=0.22,
    )

    assert close["alignment_score"] > far["alignment_score"]
