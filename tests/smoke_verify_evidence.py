from service.decision_policy import compute_graph_coupling
from scripts.benchmark.visualize_detect_benchmark import (
    PredictionRecord,
    compute_audit_summary,
)


def main() -> None:
    r1 = compute_graph_coupling(
        primary_score=0.62,
        decision_threshold=0.60,
        graph_score=0.90,
        evidence_count=0,
        base_graph_weight=0.22,
    )
    assert r1["influence_weight"] == 0.0
    assert abs(r1["coupled_score"] - 0.62) < 1e-9

    r2 = compute_graph_coupling(
        primary_score=0.62,
        decision_threshold=0.61,
        graph_score=0.90,
        evidence_count=3,
        base_graph_weight=0.22,
    )
    assert r2["influence_weight"] > 0.0
    assert r2["coupled_score"] > 0.62

    records = [
        PredictionRecord(
            path="a",
            file_name="a",
            truth_label="FAKE",
            predicted_label="FAKE",
            confidence=0.9,
            is_correct=True,
            latency_ms=1.0,
            decision_fake_score=0.81,
            decision_threshold=0.60,
            evidence_count=1,
            evidence_requested=1,
            evidence_unresolved=0,
            evidence_alignment_score=0.82,
        ),
        PredictionRecord(
            path="b",
            file_name="b",
            truth_label="FAKE",
            predicted_label="FAKE",
            confidence=0.9,
            is_correct=True,
            latency_ms=1.0,
            decision_fake_score=0.84,
            decision_threshold=0.60,
            evidence_count=0,
            evidence_requested=1,
            evidence_unresolved=0,
            evidence_alignment_score=0.0,
        ),
        PredictionRecord(
            path="c",
            file_name="c",
            truth_label="REAL",
            predicted_label="REAL",
            confidence=0.9,
            is_correct=True,
            latency_ms=1.0,
            decision_fake_score=0.21,
            decision_threshold=0.60,
            evidence_count=0,
            evidence_requested=0,
            evidence_unresolved=0,
            evidence_alignment_score=0.0,
        ),
    ]
    summary = compute_audit_summary(records)
    assert abs(summary["evidence_hit_rate"] - (1 / 3)) < 1e-9
    assert abs(summary["high_score_no_evidence_rate"] - (1 / 3)) < 1e-9
    assert summary["unresolved_subdomain_rate"] == 0.0

    print("SMOKE_ASSERT_OK")


if __name__ == "__main__":
    main()
