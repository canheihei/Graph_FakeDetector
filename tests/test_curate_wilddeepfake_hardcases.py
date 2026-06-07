from pathlib import Path

from scripts.training.curate_wilddeepfake_hardcases import (
    PredictionRow,
    is_noise_candidate,
    select_hard_cases,
)


def test_wilddeepfake_noise_and_hardcase_selection():
    rows = [
        PredictionRow(
            path=Path("Fake/a.jpg"),
            truth_label="FAKE",
            predicted_label="REAL",
            decision_fake_score=0.08,
            decision_threshold=0.10,
            is_correct=False,
        ),
        PredictionRow(
            path=Path("Fake/b.jpg"),
            truth_label="FAKE",
            predicted_label="REAL",
            decision_fake_score=0.11,
            decision_threshold=0.10,
            is_correct=False,
        ),
        PredictionRow(
            path=Path("Real/c.jpg"),
            truth_label="REAL",
            predicted_label="FAKE",
            decision_fake_score=0.88,
            decision_threshold=0.10,
            is_correct=False,
        ),
        PredictionRow(
            path=Path("Real/d.jpg"),
            truth_label="REAL",
            predicted_label="REAL",
            decision_fake_score=0.14,
            decision_threshold=0.10,
            is_correct=True,
        ),
    ]

    assert is_noise_candidate(rows[0], 0.08, 0.92) is True
    assert is_noise_candidate(rows[1], 0.08, 0.92) is False

    selected = select_hard_cases(
        rows[1:],
        hard_margin=0.05,
        hard_max_per_class=2,
        excluded_paths=set(),
    )

    assert [item.path.as_posix() for item in selected["FAKE"]] == ["Fake/b.jpg"]
    assert [item.path.as_posix() for item in selected["REAL"]] == ["Real/c.jpg", "Real/d.jpg"]
