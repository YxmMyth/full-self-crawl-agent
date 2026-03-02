# Adaptive strategy gap analysis (full_site rerun)

## Scope
- Artifacts: `reports/rerun-codepen-fullsite-e2e_20260302_115148.runlog.json`, `...stdout.log`
- Code paths: `src/pipeline.py`, `src/agents/sense.py`, `src/agents/plan.py`, `src/agents/act.py`, `src/orchestrator.py`

## 1) Why strategy auto-choice (zip vs DOM vs others) is unreliable
1. **Planner receives wrong payload shape**: pipeline passes full `sense_result` into planner as `page_structure` (`src/pipeline.py`), but planner expects a flat structure (`page_type`, `main_content_selector`, etc. in `src/agents/plan.py`), so strategy signals are often missing/misaligned.
2. **LLM planning breaks on binary data**: `SenseAgent` returns screenshot bytes (`src/agents/sense.py`), and planner does `json.dumps(page_structure)` when building selector prompt (`src/agents/plan.py`), causing `Object of type bytes is not JSON serializable`; this forces fallback selectors/strategy instead of adaptive planning.
3. **No real multi-strategy router in execution path**: runtime pipeline uses Sense→Plan→Act directly (`src/pipeline.py`); `SmartRouter` exists but is not wired into orchestrator/pipeline for method choice, so there is no robust per-site routing among SPA/API/DOM/ZIP approaches.
4. **ZIP fallback is weak heuristic**: ZIP is attempted only when extracted rows are empty/very poor (`src/agents/act.py`), using first discovered ZIP-like link only; it is not selected proactively from page signals and does not compete with other extraction methods.
5. **SPA path is effectively bypassed**: `run_spa_extraction` exists but is not used by main run path (`src/pipeline.py`), so “DOM vs API/SPA” switching is mostly absent.

## 2) What happened in latest full_site codepen run (high count, quality 0.00)
1. Run finished with `return_code=0`, `count=471`, `quality=0.00` (`...runlog.json`).
2. Logs show repeated LLM failures (`Illegal header value b'Bearer '`) and selector-planning failure (`Object of type bytes is not JSON serializable`) in many pages (`...stdout.log`), so extraction proceeded with degraded generic logic.
3. Count rose because full_site traversed many URLs and extracted generic text rows on each page, not task-aligned “ppt_html_output” artifacts.
4. Quality stayed `0.00` mainly because full_site return hardcodes `quality_score: 0.0` in orchestrator (`src/orchestrator.py`), regardless of page-level verification outcomes.

## 3) Minimal next fixes (general, not CodePen-specific)
1. **Fix planner input contract**: pass only normalized structure/features to planner (exclude screenshot/raw bytes), e.g. use `sense_result["structure"]` + selected feature fields.
2. **Make planner prompt serialization safe**: before `json.dumps`, sanitize non-JSON fields (bytes→metadata or drop) so LLM planning can run reliably.
3. **Add lightweight method router in Act/Pipeline**: select among `spa/api`, `dom/css`, and `zip` using sensed signals (`is_spa`, script-heavy + low DOM yield, zip/download anchors), and keep ZIP as fallback + optional primary when confidence high.
4. **Use real quality aggregation in full_site**: compute final quality from per-page verify results (mean or weighted) instead of hardcoded `0.0`, enabling judge/reflect feedback to influence strategy.
5. **Fail-fast on missing LLM auth**: validate API key at startup and degrade explicitly to deterministic non-LLM mode to avoid repeated noisy runtime errors.

## Expected impact
- Restores strategy observability and adaptive selection reliability across heterogeneous sites.
- Prevents inflated “count good / quality zero” outcomes by aligning extraction method and quality aggregation.
