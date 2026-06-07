from pathlib import Path


TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "templates"
    / "image-recognition.html"
)


def test_image_recognition_template_contains_defense_summary_sections():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="verdictSummaryBox"' in html
    assert 'id="topReasonsList"' in html
    assert 'id="decisionPathBox"' in html
    assert 'id="reviewGuidanceBox"' in html
    assert 'id="tracePanelsBox"' in html


def test_image_recognition_template_contains_candidate_time_and_promoted_summary_ui():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'function formatCandidateTime' in html
    assert 'function formatRelativeTime' in html
    assert 'window.promotedGroupExpandState' in html
    assert '查看详情' in html
    assert '生成于' in html
    assert '晋级于' in html


def test_image_recognition_template_allows_selecting_promoted_groups_for_delete():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'function chooseCandidateGroup' in html
    assert '选中该组' in html
    assert "if (state === 'promoted') {\n        return true;" in html
