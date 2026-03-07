# Week 7 Plan: Sentiment Agent (Phase 2) + Macro Monitor Review

**Week window**: Monday, March 16, 2026 to Sunday, March 22, 2026
**Alignment**: Phase 1 Execution Plan Section 4 (Hypercare Week 7) - "Stabilization" + Phase 2 Sentiment readiness
**Your focus**: Deliver a production-ready Sentiment Agent (dual-speed, cached, Hinglish-ready, `z_t` output) and complete full Macro Monitor indicator verification
**Partner focus**: Review evidence package, validate example outputs, and confirm Regime Agent integration handoff

---

## Week 7 Goal

Finish a working Sentiment Agent (fast lane + slow lane) while verifying Macro Monitor coverage and accuracy for all scheduled indicators. By end of week, both agents should be tested, documented, and ready for Regime Agent integration.

---

## Day-by-Day Execution

### Day 1 - Monday, March 16: Macro Monitor Full Review and Fixes

**Theme**: Validate every macro indicator path and repair data-quality gaps.

- [ ] Open Macro Monitor code and run indicator-by-indicator validation.
- [ ] Validate Beautiful Soup extraction and source correctness for:
  - CPI, WPI, IIP (official ministry sites)
  - FII/DII flows (NSE or Moneycontrol)
  - FX reserves and RBI bulletins (RBI site)
  - India-US 10Y spread (FRED or Investing.com)
  - Brent and DXY (approved source with justification)
- [ ] Fix broken scrapers and add robust error handling.
- [ ] Add or verify provenance tagging on every ingest path.
- [ ] Persist clean outputs to Silver as JSON/Parquet.

**Output**: All 10 indicators fetch successfully with accurate latest values and auditable provenance.

---

### Day 2 - Tuesday, March 17: Sentiment Agent Skeleton + Fast Lane

**Theme**: Build low-latency sentiment scoring path.

- [ ] Create `sentiment_agent.py` in Antigravity workspace.
- [ ] Implement base Sentiment Agent class with dual-speed structure (fast lane + slow lane).
- [ ] Build lightweight keyword model for intraday scoring (target <= 100 ms).
- [ ] Add Redis cache with TTL for repeated payloads.
- [ ] Use prompt in Antigravity: `Create Sentiment Agent class with fast lane using keywords and cache.`
- [ ] Test against sample text from Textual Agent output.

**Output**: Fast lane returns low-latency sentiment score with cache-backed responses.

---

### Day 3 - Wednesday, March 18: FinBERT Slow Lane + Hinglish

**Theme**: Add deep-model lane and multilingual normalization.

- [ ] Integrate FinBERT model (`ProsusAI/finbert` or equivalent).
- [ ] Implement slow lane for scheduled nightly deep scoring.
- [ ] Add Hinglish normalization strategy reused from Textual Agent.
- [ ] Incorporate Bayesian priors in slow-lane scoring logic.
- [ ] Use prompt in Antigravity: `Add FinBERT slow lane + Hinglish normalization and Bayesian priors.`
- [ ] Run on 10 real RBI/Economic Times PDFs plus X posts.

**Output**: Daily sentiment aggregates produced and ready for `z_t` computation.

---

### Day 4 - Thursday, March 19: Robustness and Quality Checks

**Theme**: Stress the sentiment pipeline and enforce quality thresholds.

- [ ] Add filters for spam, adversarial text, pump-and-dump patterns, and slang-scam patterns.
- [ ] Implement precision/recall threshold checks and timestamp alignment checks.
- [ ] Finalize cache decision policy:
  - fresh -> use
  - stale -> downweight
  - expired -> ignore
- [ ] Test with recent RBI bulletin content and hype-heavy X posts.
- [ ] Run cache-failure simulation and verify technical-only reduced-risk mode trigger.

**Output**: Robustness suite passing with documented downgrade behavior under cache failure.

---

### Day 5 - Friday, March 20: Integration with Macro Monitor + z_t

**Theme**: Connect agents and produce downstream-ready regime input.

- [ ] Connect Macro Monitor outputs into Sentiment Agent feature flow.
- [ ] Create `z_t` threshold variable using sentiment score + macro signals.
- [ ] Validate end-to-end path: Macro data + Textual records -> Sentiment Agent -> `z_t`.
- [ ] Add provenance and `quality_status` for each emitted score.
- [ ] Run full pipeline on previous-day data slice.

**Output**: End-to-end integration run with generated `z_t` values and full provenance.

---

### Day 6 - Saturday, March 21: Full Testing and Shadow Run

**Theme**: Validate both agents across multi-day historical windows.

- [ ] Run three full days of historical data through Macro Monitor and Sentiment Agent.
- [ ] Verify:
  - Macro indicators are accurate
  - Sentiment scores are sensible
  - Cache behavior is correct
  - Hinglish normalization is correct
  - `z_t` is emitted correctly
- [ ] Fix defects found during shadow run.
- [ ] Add or extend focused unit tests in `pytest`.

**Output**: Three-day shadow-run report with resolved defects and passing test evidence.

---

### Day 7 - Sunday, March 22: Documentation + Partner Review

**Theme**: Package artifacts and close Week 7 stabilization gate.

- [ ] Write concise README docs for both agents (run instructions, output schemas, `z_t` usage).
- [ ] Prepare 2-3 worked examples (for example: `CPI high + negative sentiment -> z_t = 0.85`).
- [ ] Share code and evidence package with partner for review.
- [ ] Draft next-week integration plan for Regime Agent.

**Output**: Review-ready documentation bundle, example outputs, and Regime integration next-step plan.

---

## Week 7 Exit Criteria

| Criterion | Evidence Required |
|-----------|-------------------|
| All macro indicators fetch accurately | Indicator-by-indicator validation log with source and timestamp |
| Fast lane latency target met (`<= 100 ms`) | Benchmark output on representative intraday payloads |
| Slow lane (FinBERT) operational | Daily aggregate sentiment artifact and run log |
| Hinglish normalization active | Test samples with before/after normalization traces |
| Cache policy enforced (`fresh/stale/expired`) | Cache decision logs and failure-mode test evidence |
| `z_t` computed from macro + sentiment inputs | End-to-end pipeline artifact with `z_t` values |
| Provenance + `quality_status` on all scores | Output schema validation and sample records |
| Shadow run completed (3 historical days) | Run report with pass/fail checklist and defect closure |
| Documentation and partner review complete | README links, sample outputs, and partner sign-off notes |

---

## Risk Watch for Week 7

| Risk | Mitigation |
|------|------------|
| Macro source pages change HTML structure | Keep parser fallbacks and selector-level tests per indicator |
| FinBERT runtime too slow | Use scheduled batch scoring and cache intermediate embeddings where possible |
| Hinglish false normalization | Reuse validated Textual Agent normalization rules and add regression samples |
| Cache outages degrade scoring | Trigger technical-only reduced-risk mode with explicit alerting |
| `z_t` threshold instability | Keep threshold configurable and log calibration traces for review |
