from scripts.benchmark.visualize_detect_benchmark import (
    PredictionRecord,
    compute_audit_summary,
)


def _record(
    *,
    truth_label: str,
    predicted_label: str,
    decision_fake_score: float,
    decision_threshold: float,
    evidence_count: int,
    evidence_unresolved: int = 0,
    evidence_requested: int = 0,
    evidence_alignment_score: float = 0.0,
) -> PredictionRecord:
    return PredictionRecord(
        path=f"/tmp/{truth_label.lower()}_{predicted_label.lower()}.jpg",
        file_name=f"{truth_label.lower()}_{predicted_label.lower()}.jpg",
        truth_label=truth_label,
        predicted_label=predicted_label,
        confidence=0.9,
        is_correct=truth_label == predicted_label,
        latency_ms=12.3,
        decision_fake_score=decision_fake_score,
        decision_threshold=decision_threshold,
        evidence_count=evidence_count,
        evidence_unresolved=evidence_unresolved,
        evidence_requested=evidence_requested,
        evidence_alignment_score=evidence_alignment_score,
    )


def test_audit_summary_includes_evidence_hit_rate_metrics():
    records = [
        _record(
            truth_label="FAKE",
            predicted_label="FAKE",
            decision_fake_score=0.81,
            decision_threshold=0.60,
            evidence_count=1,
            evidence_requested=1,
            evidence_alignment_score=0.82,
        ),
        _record(
            truth_label="FAKE",
            predicted_label="FAKE",
            decision_fake_score=0.84,
            decision_threshold=0.60,
            evidence_count=0,
            evidence_requested=1,
        ),
        _record(
            truth_label="REAL",
            predicted_label="REAL",
            decision_fake_score=0.21,
            decision_threshold=0.60,
            evidence_count=0,
            evidence_requested=0,
        ),
    ]

    summary = compute_audit_summary(records)

    assert summary["evidence_hit_rate"] == 1 / 3
    assert summary["high_score_no_evidence_rate"] == 1 / 3
    assert summary["unresolved_subdomain_rate"] == 0.0
    assert summary["avg_evidence_alignment_score"] > 0.0
