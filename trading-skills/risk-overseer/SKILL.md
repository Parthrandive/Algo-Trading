# SKILL: Independent Risk Overseer — Phase 4
**Project:** Multi-Agent AI Trading System — Indian Market
**Applies To:** Risk Overseer Agent · Kill Switch · Dynamic Risk Budgets · Stress Testing
**Version:** aligned with Plan v1.3.7

---

## PURPOSE
The Risk Overseer is fully independent of trading logic. It has **veto authority** over all execution decisions. Its rules take precedence over every other agent output. This skill governs kill switches, volatility-scaled risk budgets, crisis taxonomy, stress testing, and PnL attribution.

---

## 1. OPERATING MODES

| Mode | Trigger | Permitted Actions |
|------|---------|------------------|
| Normal | Default; all systems healthy | Full position opens, modifications, closes |
| Reduce-Only | Soft risk breach or feed partial failure | Close and reduce positions; no new opens |
| Close-Only | Hard feed failure or circuit breach | Close existing positions only |
| Kill-Switch | Hard limit breach or manual trigger | All order cancellation; reject new orders |

### Mode Transition Rules
- Transitions are **one-way down** (Normal → Reduce-Only → Close-Only → Kill-Switch) unless recovery criteria explicitly met
- Recovery requires: all trigger conditions resolved AND operator acknowledgement for Kill-Switch exit
- Mode change events logged with timestamp, trigger, and authorising condition

---

## 2. KILL SWITCH HIERARCHY

Layered triggers in escalation order:

| Layer | Trigger | Action |
|-------|---------|--------|
| L1 Model | Model anomaly, student drift, teacher-student divergence > threshold | Pause model; use last safe snapshot or reduce risk |
| L2 Portfolio | Max drawdown breach, daily loss limit, position concentration breach | Enter Reduce-Only; alert operator |
| L3 Broker | Broker API error, margin call, rejection rate > threshold | Enter Close-Only; alert operator immediately |
| L4 Manual | Operator command via dashboard or runbook | Immediate Kill-Switch; log operator ID |

- All kill switch drills must pass in Phase 4 gate review
- Drill schedule: monthly for L1–L3; quarterly full-stack including L4

---

## 3. DYNAMIC VOLATILITY-SCALED RISK BUDGETS (Tier 1-B)

### 3.1 Volatility Regime Mapping
Define sigma thresholds per asset cluster (document in config):

| Regime | Realised Vol Condition | Exposure Cap |
|--------|----------------------|--------------|
| Normal | vol <= 1× σ_baseline | 100 % of normal budget |
| Elevated | 1× < vol <= 1.5× σ_baseline | 70 % of normal budget |
| High | 1.5× < vol <= 2× σ_baseline | 40 % of normal budget |
| Extreme | vol > 2× σ_baseline | 15 % of normal budget (protective) |

- σ_baseline is defined per asset cluster and reviewed quarterly
- Cap adjustment is **automatic** and fires **before** hard kill-switch thresholds
- Cap transitions are deterministic and auditable; every transition emits `RISK_CAP_CHANGE` event

### 3.2 False-Trigger Rate Control
- Track false-trigger rate on rolling 30-day window
- Alert if false-trigger rate exceeds rollout acceptance limit (define limit in config)
- If false-trigger rate breaches limit: pause auto-adjustment; escalate to manual review

### 3.3 Risk Telemetry (Required)
- Emit regime state, exposure cap level, and cap-change events to observability dashboard
- Alerts: regime transition, cap reduction >= 2 steps, repeated false triggers

---

## 4. CRISIS TAXONOMY

| State | Definition | Mode Transition | Recovery |
|-------|-----------|-----------------|----------|
| Full Crisis | All 3 entry conditions met (vol break + liquidity deterioration + agent confidence drop) | Close-Only; crisis-weighted consensus | Multi-condition revalidation required |
| Agent Divergence | >= 2 major agents fundamentally disagree on direction | Neutral-Hold (freeze new opens) | >= 2 consecutive aligned signals + hold duration elapsed |
| Slow-Crash / Freeze | Gradual drawdown accumulation or feed freeze without flash shock | Staged de-risking; do not wait for full crisis confirmation | Risk metrics return to normal threshold |

### 4.1 Crisis Entry Protocol (Mandatory)
1. Confirm all 3 conditions true simultaneously
2. Conditions must persist for **configured ticks/seconds** (hysteresis; define in config)
3. Log `CRISIS_ENTRY` with all 3 condition values and timestamps
4. Apply crisis-weighted routing: crisis agent weight <= 70 %; **never 100 %**
5. Set max-duration timer; auto-revert if conditions not revalidated at expiry

### 4.2 Agent Divergence Protocol
- Neutral-hold duration: configurable (e.g., 15–30 minutes); document in config
- Staged re-risking: 25 % → 50 % → 75 % → 100 % of normal budget, one step per confirmed alignment signal
- Re-risking steps cannot be skipped

---

## 5. RISK CONTROLS & LIMITS

### 5.1 Position-Level Controls
- Max position size per instrument: defined per asset bucket
- Max concentration: single stock / sector / asset class limits enforced at order submission
- Correlation limits: cross-asset correlation checked before each new open

### 5.2 Portfolio-Level Controls
- Max drawdown limit: Hard stop (define in config; go-live target: MDD <= 8 %)
- Daily loss limit: Hard stop at configurable percentage of NAV
- Weekly review of capacity utilisation vs ₹5–10 Cr initial ceiling

### 5.3 Negative Sentiment Protocol
- Extreme negative sentiment spike (z_t < configurable threshold) → trigger protective action
- Sentiment-to-price mismatch (sentiment positive, price rapidly falling) → downgrade sentiment to neutral; increase hedge exposure

### 5.4 OOD / Alien-State Staged De-Risking
- OOD flag → do not immediately neutralise; follow staged protocol (see Regime Agent Skill Section 2.3)
- Exception: if OOD coincides with hard limit breach, bypass staging → immediate Kill-Switch

---

## 6. STRESS TESTING FRAMEWORK

### 6.1 Mandatory Scenario Library
All scenarios must be versioned, labelled, and reviewed quarterly:

| Scenario | Type | Key Test |
|----------|------|----------|
| RBI surprise rate hike | Historical analogue | INR move, bond yield spike |
| INR flash move | Historical + synthetic | Execution continuity, hedge trigger speed |
| Liquidity drought | Synthetic | Impact cost spike, participation limit breach |
| 2008 GFC data | Historical | Portfolio survival, drawdown containment |
| 2013 Taper Tantrum | Historical | Regime transition speed, macro response |
| COVID crash (Feb–Apr 2020) | Historical | Multi-asset correlation behaviour |
| Correlation inversion | Synthetic (IMPOSSIBLE) | Risk model coherence under contradiction |
| Frozen constituent prices | Synthetic (IMPOSSIBLE) | Zero-volume vs missing-data distinction |
| Multi-asset liquidity vacuum | Synthetic (IMPOSSIBLE) | Safe mode activation |
| Data poisoning / feed freeze | Synthetic | Correct degraded-mode behaviour |

### 6.2 Capacity Stress Tests
Run every scenario at **1×, 2×, and 3× current notional capacity**:
- Hard cap at 2× if impact > 0.25 % at 3×
- Log results and update capacity limits if 3× tests fail

### 6.3 Impossible-Scenario Requirements
- Validate engine distinguishes **zero volume** from **missing data**
- Engine must enter safe mode when feed integrity is uncertain (not just when feed is missing)
- Correlation inversion: risk model must not produce undefined/negative variance

### 6.4 Snapback Tests
- Measure: how many ticks to clip net exposure after flash-shock
- Log results; do **not** auto-tune smoothing parameters from snapback output (manual review only)
- Alert if snapback ticks exceed configured threshold

### 6.5 Quarterly Stress Review
- All scenarios rerun; results compared to prior quarter
- Trigger risk budget or capacity adjustments if degradation detected
- Review signed by owner and reviewer

---

## 7. XAI & ATTRIBUTION

### 7.1 SHAP Logging Requirements
- Log top-k feature contributions and agent contributions for every trade
- Coverage target: >= 80 % of trades with top-k explanations logged
- XAI logs retained and queryable for post-trade analysis

### 7.2 Live PnL Attribution
- PnL attribution available per agent and per signal family in real time
- Supports diagnosis during stress windows without requiring post-hoc analysis
- Attribution dashboard panel active in production

---

## 8. DRIFT & ANOMALY DETECTION
- ADDM or equivalent automated drift detection running continuously
- Drift alert triggers model review; sustained drift triggers demotion process
- Data provenance reliability score dynamically adjusts position sizing (not fixed haircut rules)

---

## RISK OVERSEER CHECKLIST
- [ ] Kill-switch hierarchy documented; all 4 layers drill-tested
- [ ] Dynamic vol-scaled budgets: sigma thresholds defined per asset cluster; false-trigger rate monitored
- [ ] Crisis taxonomy complete: full crisis, divergence, slow-crash all have documented mode transitions
- [ ] Crisis entry: 3-condition confirmation, hysteresis ticks configured, max-duration timer set
- [ ] Stress scenario library versioned; all scenarios labelled (historical vs synthetic vs impossible)
- [ ] Capacity stress: 1×/2×/3× tests completed; hard cap at 2× enforced if 3× fails
- [ ] SHAP coverage >= 80 % of trades
- [ ] Live PnL attribution dashboard active
- [ ] Drift detection running; demotion process tested
- [ ] Quarterly stress review scheduled with named owner
