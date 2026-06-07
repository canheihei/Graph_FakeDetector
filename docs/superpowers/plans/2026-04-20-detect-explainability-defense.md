# Detect Explainability Defense Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework `/detect` into a conclusion-first, defense-friendly explanation flow without breaking the existing audit contract.

**Architecture:** Add a backend explanation-summary builder inside `DetectionFacade`, then refactor the detect page to consume that summary as the primary display while keeping raw audit sections available behind expandable panels. Tests lock both the backend summary contract and the new template structure.

**Tech Stack:** Flask, Python, server-rendered HTML, vanilla JavaScript, Tailwind CDN styling

---

### Task 1: Add Backend Regression Test For Detect Explanation Summary

**Files:**
- Create: `tests/test_detect_explain_summary.py`
- Modify: `service/facades.py`
- Test: `tests/test_detect_explain_summary.py`

- [ ] **Step 1: Write the failing test**

```python
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
        diagnostic_chain=["a", "b", "c"],
    )

    assert summary["verdict_summary"]["title"] == "判定为疑似伪造"
    assert len(summary["top_reasons"]) == 3
    assert any("图谱" in item for item in summary["top_reasons"])
    assert "图谱证据参与" in summary["decision_path"]["summary"]
    assert summary["review_summary"]["needs_review"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detect_explain_summary.py -v`
Expected: FAIL because `_build_explain_summary` does not exist or does not return the required structure.

- [ ] **Step 3: Write minimal implementation**

```python
def _build_explain_summary(...):
    return {
        "verdict_summary": {...},
        "top_reasons": [...],
        "decision_path": {...},
        "review_summary": {...},
        "trace_panels": {...},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_detect_explain_summary.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_detect_explain_summary.py service/facades.py
git commit -m "feat: add detect explain summary payload"
```

### Task 2: Wire Explanation Summary Into `/detect`

**Files:**
- Modify: `service/facades.py`
- Test: `tests/test_detect_explain_summary.py`

- [ ] **Step 1: Write the failing integration-style test**

```python
def test_detect_response_includes_explain_summary():
    response = facade.execute(request)
    assert "explain_summary" in response
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detect_explain_summary.py -v`
Expected: FAIL because `execute()` does not include `explain_summary`.

- [ ] **Step 3: Write minimal implementation**

```python
explain_summary = self._build_explain_summary(...)
return {
    ...
    "explain_summary": explain_summary,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_detect_explain_summary.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add service/facades.py tests/test_detect_explain_summary.py
git commit -m "feat: expose detect explain summary"
```

### Task 3: Add Detect Template Regression Test

**Files:**
- Create: `tests/test_image_recognition_template.py`
- Modify: `frontend/templates/image-recognition.html`
- Test: `tests/test_image_recognition_template.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_image_recognition_template_contains_defense_summary_sections():
    html = Path("frontend/templates/image-recognition.html").read_text(encoding="utf-8")

    assert 'id="verdictSummaryBox"' in html
    assert 'id="topReasonsList"' in html
    assert 'id="decisionPathBox"' in html
    assert 'id="reviewGuidanceBox"' in html
    assert 'id="tracePanelsBox"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_image_recognition_template.py -v`
Expected: FAIL because the new summary containers are missing.

- [ ] **Step 3: Write minimal implementation**

```html
<div id="verdictSummaryBox"></div>
<ul id="topReasonsList"></ul>
<div id="decisionPathBox"></div>
<div id="reviewGuidanceBox"></div>
<div id="tracePanelsBox"></div>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_image_recognition_template.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_image_recognition_template.py frontend/templates/image-recognition.html
git commit -m "feat: add detect defense layout containers"
```

### Task 4: Rebuild Detect Page Around Conclusion-First Layout

**Files:**
- Modify: `frontend/templates/image-recognition.html`
- Test: `tests/test_image_recognition_template.py`

- [ ] **Step 1: Reorganize the result card markup**

```html
<div class="space-y-6">
  <section id="verdictSummaryBox"></section>
  <section id="topReasonsBox"></section>
  <section id="decisionPathBox"></section>
  <section id="reviewGuidanceBox"></section>
  <section id="tracePanelsBox"></section>
</div>
```

- [ ] **Step 2: Update the client-side render logic to use `data.explain_summary`**

```javascript
const explain = data.explain_summary || {};
renderVerdictSummary(explain.verdict_summary || {});
renderTopReasons(explain.top_reasons || []);
renderDecisionPath(explain.decision_path || {});
renderReviewGuidance(explain.review_summary || {});
renderTracePanels(explain.trace_panels || {});
```

- [ ] **Step 3: Keep existing evidence/raw audit rendering inside expandable sections**

```javascript
tracePanelsBox.innerHTML = `
  <details>...</details>
  <details>...</details>
  <details>...</details>
`;
```

- [ ] **Step 4: Run template regression test**

Run: `python -m pytest tests/test_image_recognition_template.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/templates/image-recognition.html tests/test_image_recognition_template.py
git commit -m "feat: redesign detect page for defense explainability"
```

### Task 5: Verify End-To-End Behavior

**Files:**
- Modify: `AGENTS.md`
- Test: `tests/test_detect_explain_summary.py`
- Test: `tests/test_image_recognition_template.py`

- [ ] **Step 1: Update project notes**

```markdown
- [Done] detect 页面改为结论优先的答辩展示，新增 explain_summary 与复核指引
```

- [ ] **Step 2: Run backend regression test in the cloud detector environment**

Run:

```bash
ssh -p 49649 root@ae2836a105e54a59892c240731db2e15.region1.waas.aigate.cc \
  "cd /root/pycode/graph_detect && /root/miniconda3/bin/conda run -n detector --no-capture-output python -m pytest tests/test_detect_explain_summary.py -v"
```

Expected: PASS

- [ ] **Step 3: Run template regression checks**

Run:

```bash
python -m pytest tests/test_image_recognition_template.py -v
```

Expected: PASS, or if local Python is unavailable, run an equivalent file-content assertion.

- [ ] **Step 4: Verify no JavaScript syntax errors in the detect page**

Run:

```bash
node -e "const fs=require('fs'); const html=fs.readFileSync('frontend/templates/image-recognition.html','utf8'); const m=html.match(/<script>([\s\S]*)<\/script>/); new Function(m[1]); console.log('JS_PARSE_PASS');"
```

Expected: `JS_PARSE_PASS`

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md service/facades.py frontend/templates/image-recognition.html tests/test_detect_explain_summary.py tests/test_image_recognition_template.py
git commit -m "feat: improve detect explainability for defense"
```
