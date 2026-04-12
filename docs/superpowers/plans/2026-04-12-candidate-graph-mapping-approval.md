# Candidate Graph And Mapping Approval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a candidate-layer workflow where weak-evidence fake detections can generate LLM-backed graph and mapping candidates, review them in the iterate UI, benchmark them, and promote only approved candidates into `alignment/mapping_config.json`.

**Architecture:** Keep detect’s active evidence path stable while introducing a separate candidate persistence layer. Candidate graph structure is stored in Neo4j, candidate mapping approval state is stored in `alignment/mapping_candidates.json`, and benchmarks apply candidates through an in-memory overlay instead of mutating the active mapping file.

**Tech Stack:** Flask, Pydantic, Neo4j, existing detector/detect facade pipeline, Tailwind template pages, pytest.

---

### Task 1: Add Candidate Storage Models

**Files:**
- Create: `service/candidate_store.py`
- Create: `tests/test_candidate_store.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from service.candidate_store import CandidateStore


def test_candidate_store_round_trips_items(tmp_path: Path):
    store = CandidateStore(tmp_path / "mapping_candidates.json")
    payload = {
        "candidate_id": "c1",
        "status": "pending",
        "graph_candidate": {"specific_domain": "后处理痕迹域"},
        "mapping_candidate": {"detector": "FFTDetector", "feature": "patch_inconsistency"},
    }

    store.append_items([payload])
    items = store.list_items()

    assert len(items) == 1
    assert items[0]["candidate_id"] == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_store.py::test_candidate_store_round_trips_items -v`
Expected: FAIL with `ModuleNotFoundError` for `service.candidate_store`

- [ ] **Step 3: Write minimal implementation**

```python
import json
from pathlib import Path


class CandidateStore:
    def __init__(self, path: Path):
        self._path = Path(path)

    def _read(self):
        if not self._path.exists():
            return {"version": "1.0", "items": []}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, payload):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_items(self):
        return list(self._read().get("items", []))

    def append_items(self, items):
        payload = self._read()
        payload.setdefault("items", []).extend(items)
        self._write(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_store.py::test_candidate_store_round_trips_items -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add service/candidate_store.py tests/test_candidate_store.py
git commit -m "feat: add candidate mapping json store"
```

### Task 2: Add Candidate Eligibility And Prompt Parsing

**Files:**
- Create: `service/candidate_generation.py`
- Create: `tests/test_candidate_generation.py`
- Modify: `service/llm_chain.py`
- Modify: `project_paths.py`

- [ ] **Step 1: Write the failing test**

```python
from service.candidate_generation import should_generate_candidates


def test_should_generate_candidates_for_fake_model_only_detection():
    detect_result = {
        "label": "FAKE",
        "reasoning_type": "anomaly_model_only",
        "evidence": [],
        "evidence_diagnostics": {"unresolved_subdomains": 0},
    }

    assert should_generate_candidates(detect_result) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_generation.py::test_should_generate_candidates_for_fake_model_only_detection -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
def should_generate_candidates(detect_result: dict) -> bool:
    if str(detect_result.get("label", "")) != "FAKE":
        return False
    if str(detect_result.get("reasoning_type", "")) == "anomaly_model_only":
        return True
    if len(detect_result.get("evidence", []) or []) == 0:
        return True
    diagnostics = detect_result.get("evidence_diagnostics", {}) or {}
    return int(diagnostics.get("unresolved_subdomains", 0) or 0) > 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_generation.py::test_should_generate_candidates_for_fake_model_only_detection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add service/candidate_generation.py tests/test_candidate_generation.py project_paths.py service/llm_chain.py
git commit -m "feat: add candidate eligibility and llm parsing"
```

### Task 3: Persist Candidate Graph In Neo4j

**Files:**
- Modify: `service/neo_client.py`
- Create: `tests/test_candidate_graph_queries.py`

- [ ] **Step 1: Write the failing test**

```python
from service.neo_client import build_candidate_subdomain_params


def test_build_candidate_subdomain_params_keeps_candidate_metadata():
    params = build_candidate_subdomain_params(
        candidate_id="c1",
        graph_candidate={"specific_domain": "后处理痕迹域", "subdomain_name": "边缘重采样失真"},
        source={"sample_name": "dfdc_fake_0001.jpg"},
    )

    assert params["candidate_id"] == "c1"
    assert params["subdomain_name"] == "边缘重采样失真"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_graph_queries.py::test_build_candidate_subdomain_params_keeps_candidate_metadata -v`
Expected: FAIL because helper does not exist

- [ ] **Step 3: Write minimal implementation**

```python
def build_candidate_subdomain_params(*, candidate_id, graph_candidate, source):
    return {
        "candidate_id": candidate_id,
        "specific_domain": graph_candidate["specific_domain"],
        "subdomain_name": graph_candidate["subdomain_name"],
        "sample_name": source.get("sample_name", ""),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_graph_queries.py::test_build_candidate_subdomain_params_keeps_candidate_metadata -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add service/neo_client.py tests/test_candidate_graph_queries.py
git commit -m "feat: add candidate graph persistence helpers"
```

### Task 4: Add Candidate Generation Endpoint

**Files:**
- Modify: `app.py`
- Modify: `service/facades.py`
- Create: `tests/test_candidate_api.py`

- [ ] **Step 1: Write the failing test**

```python
from service.facades import CandidateRequest


def test_candidate_request_accepts_detect_context():
    request = CandidateRequest(
        detect_result={"label": "FAKE"},
        source_sample_name="sample.jpg",
    )

    assert request.source_sample_name == "sample.jpg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_api.py::test_candidate_request_accepts_detect_context -v`
Expected: FAIL because `CandidateRequest` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateRequest:
    detect_result: dict
    source_sample_name: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_api.py::test_candidate_request_accepts_detect_context -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py service/facades.py tests/test_candidate_api.py
git commit -m "feat: add candidate generation api"
```

### Task 5: Add Benchmark Overlay Service

**Files:**
- Create: `service/candidate_benchmark.py`
- Create: `tests/test_candidate_benchmark.py`
- Modify: `alignment/aligner.py`
- Modify: `service/facades.py`

- [ ] **Step 1: Write the failing test**

```python
from service.candidate_benchmark import merge_mapping_rules


def test_merge_mapping_rules_replaces_rule_for_same_detector_feature():
    active = {"version": "1.4", "rules": [{"detector": "FFTDetector", "feature": "patch_inconsistency", "subdomain_label": "旧节点"}]}
    candidate = {"mapping_candidate": {"detector": "FFTDetector", "feature": "patch_inconsistency", "subdomain_label": "新节点"}}

    merged = merge_mapping_rules(active, [candidate])

    assert merged["rules"][0]["subdomain_label"] == "新节点"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_benchmark.py::test_merge_mapping_rules_replaces_rule_for_same_detector_feature -v`
Expected: FAIL because helper does not exist

- [ ] **Step 3: Write minimal implementation**

```python
def merge_mapping_rules(active_config: dict, candidates: list[dict]) -> dict:
    rules = {}
    for rule in active_config.get("rules", []):
        rules[(rule["detector"], rule["feature"])] = dict(rule)
    for candidate in candidates:
        rule = dict(candidate["mapping_candidate"])
        rules[(rule["detector"], rule["feature"])] = rule
    return {"version": active_config.get("version", "1.0"), "rules": list(rules.values())}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_benchmark.py::test_merge_mapping_rules_replaces_rule_for_same_detector_feature -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add service/candidate_benchmark.py tests/test_candidate_benchmark.py alignment/aligner.py service/facades.py
git commit -m "feat: add candidate benchmark overlay service"
```

### Task 6: Add Detect-Side Review UI

**Files:**
- Modify: `frontend/templates/image-recognition.html`
- Modify: `frontend/templates/graph-iteration.html`

- [ ] **Step 1: Write the failing test**

There is no frontend test harness in this repo. Create a manual verification checklist in the code comments for this task before implementation:

```html
<!-- Manual verification:
1. Run detect on a weak-evidence fake sample.
2. Click "生成候选" on image-recognition page.
3. Confirm grouped candidate alternatives render on the same detect page.
4. Use group radios, bulk select, run quick benchmark, then promote. -->
```

- [ ] **Step 2: Run manual static verification**

Open the template after the placeholder comment is added and confirm the checklist is visible in source.

- [ ] **Step 3: Write minimal implementation**

Add:

- A detect-page candidate review panel with grouped radio selection, bulk select, inline editable inputs, and action buttons for quick/formal benchmark and promote
- A detect-page operation progress bar for benchmark/promote
- Remove duplicated candidate review surface from iterate page and leave a short redirect note

- [ ] **Step 4: Run manual verification**

Run the Flask app in the intended environment and verify the four manual checklist items.

- [ ] **Step 5: Commit**

```bash
git add frontend/templates/graph-iteration.html frontend/templates/image-recognition.html
git commit -m "feat: add candidate approval ui"
```

### Task 7: Update Docs And Ops Notes

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write the failing test**

Documentation task. Define an explicit checklist in the doc patch:

```markdown
- Candidate JSON file path documented
- Candidate API endpoints documented
- Promotion workflow documented
- Quick/formal benchmark workflow documented
```

- [ ] **Step 2: Verify docs are missing those items**

Run: `rg -n "mapping_candidates|/detect/candidates|candidate-mappings/promote" README.md AGENTS.md`
Expected: no matches before the patch

- [ ] **Step 3: Write minimal implementation**

Document:

- new candidate endpoints
- candidate approval workflow
- cloud verification notes

- [ ] **Step 4: Run verification**

Run: `rg -n "mapping_candidates|/detect/candidates|candidate-mappings/promote" README.md AGENTS.md`
Expected: matches appear in both docs

- [ ] **Step 5: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: describe candidate mapping approval workflow"
```

## Self-Review

- Spec coverage: tasks cover storage, generation, persistence, benchmark overlay, UI, and docs. No spec section is intentionally omitted.
- Placeholder scan: there are no `TODO` placeholders; frontend validation is explicitly called out as manual because the repo has no UI test harness.
- Type consistency: `CandidateRequest`, `CandidateStore`, and `merge_mapping_rules` names are consistent across tasks.

Plan complete and saved to `docs/superpowers/plans/2026-04-12-candidate-graph-mapping-approval.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
