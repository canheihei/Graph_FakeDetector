# Detect Explainability Defense Design

## Context

Current `detect` output is technically complete but defense-hostile. Teachers struggle to answer two core questions from the UI:

1. Why was this image judged `FAKE` or `REAL`?
2. If the result is uncertain, how can a human review it?

The current `image-recognition.html` emphasizes evidence lists and audit fields, but not a clear narrative. The evidence-chain report page further focuses on aggregate metrics rather than reinforcing the single-sample explanation flow.

## Goal

Redesign `detect` presentation so the default experience is conclusion-first and defense-oriented, while preserving the existing audit fields and interface stability.

## Product Direction

The presentation should feel like a product demo, not a debug console:

- Default view: simple, direct, human-readable
- Expanded view: structured evidence
- Deepest view: raw audit trail for traceability

## Scope

This design only changes the `detect` experience:

- Backend: add a defense-readable explanation summary to the existing `/detect` response
- Frontend: reorganize `image-recognition.html` around conclusion, top reasons, review guidance, and expandable trace sections
- Keep existing audit fields such as `reasoning_type`, `diagnostic_chain`, `risk_level`, `needs_review`, and `evidence_diagnostics`

Out of scope:

- Redesigning `iterate`
- Reworking benchmark logic
- Changing `/detect` decision logic itself
- Rebuilding `evidence-chain-report.html` in this implementation slice

## User Experience

### Layer 1: Immediate Verdict

The first visible result block must answer, within a few seconds:

- What did the system decide?
- How confident is it?
- Should a human review it?

This layer should show:

- Verdict label
- Confidence
- Review badge
- One-sentence summary

### Layer 2: Three Key Reasons

The default explanation should contain at most three short reasons written in plain Chinese. These reasons should be synthesized from existing detector, evidence, and decision fields instead of exposing internal variable names.

Examples:

- Multiple detector signals jointly indicate forged anomalies
- Graph evidence hit stable forgery-related subdomains
- The score is clearly above the active decision threshold

### Layer 3: Decision Path

Show a simplified causal path:

`输入图像 -> 检测器发现异常 -> 图谱证据参与/未参与 -> 最终结论`

The path must explain whether the graph actually participated, because that is a common defense challenge.

### Layer 4: Human Review Guidance

If review is needed, the UI must state:

- Why review is needed
- What the reviewer should focus on first

This converts the existing risk/audit mechanism into something defense audiences can trust.

### Expanded Evidence Area

Below the summary layer, add collapsible sections:

- 模型证据
- 图谱证据
- 决策与风险
- 完整可追溯记录

The raw audit content remains available but is no longer the default entry point.

## Backend Design

Add a new response object, tentatively `explain_summary`, to `/detect`. It should be derived from the existing decision and audit fields without changing the meaning of those original fields.

Suggested structure:

- `verdict_summary`
- `top_reasons`
- `decision_path`
- `review_summary`
- `trace_panels`

The summary builder belongs in `DetectionFacade`, close to the existing reasoning/audit generation, so the UI does not have to reconstruct meaning from low-level fields.

## Frontend Design

`frontend/templates/image-recognition.html` should be reorganized into:

1. Verdict hero
2. Top reasons
3. Decision path
4. Review guidance
5. Expandable evidence sections

The existing evidence list and full chain box should be preserved as part of the expanded sections rather than removed.

## Testing Strategy

Use TDD:

- Add backend tests for explanation summary generation
- Add template regression checks for new detect-page containers and labels
- Verify that existing audit fields still exist in the response contract

Python-dependent verification should prefer the cloud `detector` environment.

## Risks

- `image-recognition.html` is already large; implementation should avoid uncontrolled complexity
- Explanation text can become too verbose or too generic; keep it concise and deterministic
- The summary must not contradict the original audit fields

## Acceptance Criteria

- A teacher can understand the system verdict from the top result area without reading raw evidence fields
- The page explicitly states whether graph evidence participated
- The page explicitly states whether human review is recommended and why
- Original audit fields remain available for traceability
- `/detect` interface compatibility is preserved
