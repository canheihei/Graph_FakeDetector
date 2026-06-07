from service.facades import DetectionFacade


def test_build_explain_summary_highlights_fake_with_graph_evidence():
    facade = DetectionFacade.__new__(DetectionFacade)

    summary = facade._build_explain_summary(
        decision={
            "label": "FAKE",
            "confidence": 0.94,
            "decision_fake_score": 0.81,
            "decision_threshold": 0.48,
            "decision_margin": 0.33,
            "evidence_alignment_score": 0.77,
            "graph_influence_weight": 0.28,
        },
        evidence=[
            {
                "sub_domain": {"name": "边界融合不连续"},
                "specific_domain": {"name": "后处理痕迹域"},
                "confidence": 0.83,
            }
        ],
        reasoning_type="anomaly_evidence",
        needs_review=False,
        review_reasons=[],
        risk_level="none",
        detector_signals=[
            {"name": "FFTDetector:high_freq_energy", "score": 0.86, "weight": 0.9},
            {"name": "BoundaryDetector:blend_border", "score": 0.74, "weight": 0.8},
        ],
        evidence_diagnostics={"requested_subdomains": 2, "unresolved_subdomains": 0},
        diagnostic_chain=["Input analyzed", "Graph evidence activated", "Fake verdict emitted"],
    )

    assert summary["verdict_summary"]["title"] == "判定为疑似伪造"
    assert summary["verdict_summary"]["review_badge"] == "可直接采信"
    assert len(summary["top_reasons"]) == 3
    assert any("图谱" in item for item in summary["top_reasons"])
    assert "图谱证据参与" in summary["decision_path"]["summary"]
    assert summary["review_summary"]["needs_review"] is False
    assert "完整可追溯记录" in summary["trace_panels"]


def test_build_explain_summary_marks_review_when_fake_has_no_graph_evidence():
    facade = DetectionFacade.__new__(DetectionFacade)

    summary = facade._build_explain_summary(
        decision={
            "label": "FAKE",
            "confidence": 0.61,
            "decision_fake_score": 0.52,
            "decision_threshold": 0.48,
            "decision_margin": 0.04,
            "evidence_alignment_score": 0.0,
            "graph_influence_weight": 0.0,
        },
        evidence=[],
        reasoning_type="anomaly_model_only",
        needs_review=True,
        review_reasons=["fake_without_graph_evidence", "near_decision_boundary"],
        risk_level="medium",
        detector_signals=[
            {"name": "CalibratedVision:fake_probability", "score": 0.52, "weight": 1.0},
        ],
        evidence_diagnostics={"requested_subdomains": 1, "unresolved_subdomains": 1},
        diagnostic_chain=["Input analyzed", "No graph evidence", "Boundary fake verdict emitted"],
    )

    assert summary["verdict_summary"]["title"] == "判定为疑似伪造"
    assert summary["verdict_summary"]["review_badge"] == "建议人工复核"
    assert "未命中稳定图谱证据" in summary["decision_path"]["summary"]
    assert summary["review_summary"]["needs_review"] is True
    assert any("人工复核" in item for item in summary["review_summary"]["review_reasons_human"])
