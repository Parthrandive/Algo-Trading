# SKILL: Execution & Fast Loop Discipline
**Project:** Multi-Agent AI Trading System — Indian Market
**Applies To:** Strategic Executive (Trader Ensemble) · Fast Loop · Slow Loop · Order Routing
**Version:** aligned with Plan v1.3.7

---

## PURPOSE
This skill governs the execution-critical path — the Fast Loop — and the asynchronous Slow Loop. It enforces latency discipline, teacher-student inference rules, order routing behaviour, and market impact controls. Violations of these rules can cause live trading losses; all rules are non-negotiable.

---

## 1. DUAL-LOOP ARCHITECTURE

### 1.1 Fast Loop (Execution Critical)
| Parameter | Requirement |
|-----------|-------------|
| Latency target | p99 <= **8 ms** (stretch goal) |
| Enforced degrade threshold | p99 > **10 ms** → automatic degrade or bypass |
| Monitoring | p50, p95, p99, p99.9, and jitter instrumented at every stage |
| Allowed inputs | Technical microstructure features, order-book imbalance snapshots, cache reads (sentiment, regime) |
| Inference allowed | **Student policy only** — distilled, deterministic, pre-approved artifact |
| Inference blocked | Teacher policy, heavy NLP, fresh sentiment inference, simulation |
| Cache reads | O(1) memory reads only; no blocking I/O |

### 1.2 Slow Loop (Asynchronous Context & Policy)
| Parameter | Requirement |
|-----------|-------------|
| Typical latency | 100–500 ms |
| Blocking | **NEVER** blocks order placement |
| Role | Regime context updates, teacher policy inference, policy refresh, heavy analytics |
| Policy refresh | Every 10 minutes scheduled, or earlier on async completion |
| Output | Latest validated policy snapshot consumed by Fast Loop |

### 1.3 Loop Separation Rules
- Slow Loop outputs are **never** pulled synchronously inside the Fast Loop
- Fast Loop reads only from pre-written snapshots/caches updated by Slow Loop
- Any computation that exceeds 5 ms budget must be moved to Slow Loop
- Volatility bypass triggers use hysteresis + cooldown; no rapid mode flapping permitted

---

## 2. TEACHER-STUDENT INFERENCE

### 2.1 Student Policy Requirements
Before any student policy is promoted to production:
- [ ] Deterministic runtime profile verified (same input → same output, always)
- [ ] Approved compression plan documented: quantisation / pruning / distillation method and parameters
- [ ] p99 and p99.9 tail-latency gates pass on worst-case replay
- [ ] Teacher-student agreement threshold passes on **crisis slices** (not just average-day data)
- [ ] Rollback path tested and documented

### 2.2 Teacher Policy Rules
- Teacher inference is **blocked from the execution-critical path at all times**
- Teacher runs offline or in Slow Loop only
- Live production analytics monitor teacher output; student drift beyond threshold triggers automatic demotion
- Student drift definition: configurable agreement-rate threshold on rolling window; document threshold in config

### 2.3 Ensemble Composition (Reference)
| Policy | Role |
|--------|------|
| SAC (Soft Actor-Critic) | Primary; regime-switch robustness via max-entropy framework |
| PPO (Proximal Policy Optimisation) | Challenger / diversity |
| TD3 (Twin Delayed DDPG) | Challenger / diversity |
| Market-making head | **Default OFF**; deferred to Tier 3; never enable in current cycle |

---

## 3. ORDER-BOOK IMBALANCE IN FAST LOOP (Tier 1-C)

### 3.1 Feature Schema
Define canonical imbalance features:
- `bid_ask_imbalance`: (bid_qty – ask_qty) / (bid_qty + ask_qty) at top N levels
- `queue_pressure`: signed order-arrival rate differential
- Schema version must be incremented on any field change

### 3.2 Feature Quality Guardrails
| Condition | Action |
|-----------|--------|
| Feature age > staleness threshold | Mark `STALE`; degrade gracefully |
| Missing book levels | Mark `INCOMPLETE`; use last good snapshot |
| Source quality flag = DEGRADED | Downweight imbalance signal; do not block execution |
| Fast Loop p99 breached after enablement | Disable imbalance features; escalate |

### 3.3 Safety Rule
Feature quality failure must **never block execution**. Degrade gracefully: fall back to non-imbalance signal set and log `IMBALANCE_DEGRADED` event.

---

## 4. MARKET IMPACT & SLIPPAGE CONTROLS (Tier 1-A)

### 4.1 Impact Threshold Buckets
Define thresholds by instrument bucket:
| Bucket | Examples | Participation Limit | Impact Alert Threshold |
|--------|----------|--------------------|-----------------------|
| Liquid Large-Cap | Nifty 50 constituents, Bank Nifty | Defined at design time | Defined at design time |
| Mid-Liquidity | Nifty Midcap100 names | Defined at design time | Higher threshold |
| USD/INR Futures | Currency segment | Defined at design time | FX-specific |
| Gold Futures (MCX) | GOLD, GOLDM | Defined at design time | COMEX-linked adjustment |

### 4.2 Real-Time Impact Monitor
- Track per-order: realised slippage, participation rate, ADV-linked impact score
- Compare realised slippage vs model estimate; alert if realised > model + **20 bps**
- Track at fill level, not just order level

### 4.3 Auto-Reduction Protocol
On impact threshold breach:
1. Emit `IMPACT_BREACH` event with instrument, threshold, and realised value
2. Reduce position sizing by configured step (e.g., 25 %) immediately
3. Apply cooldown window before next size restoration
4. Hysteresis: require 2 consecutive clean fills before restoring full sizing
5. Log all auto-reduction events to dashboard

### 4.4 Execution Quality Metrics (Required)
- Fill rate vs order intent
- Slippage: realised vs model estimate (bps)
- Decision-to-fill latency (p50, p99)
- Participation rate vs ADV
- Circuit breaker / halt encounters: log and report daily

### 4.5 Capacity Limits (Hard)
- Live notional: ₹5–10 Cr initial; scale to ₹25 Cr only after 6 months live validation
- Hard cap: 2× initial if 3× stress test shows impact > 0.25 %
- Max participation limits enforced at execution time; broker-side pre-trade check mandatory

---

## 5. LATENCY DISCIPLINE (Tier 1-D)

### 5.1 Instrumentation Requirements
Instrument the full decision path at every stage:
- Feed ingest → normalisation → feature computation → cache read → student inference → order routing
- Emit p50, p95, p99, p99.9, and jitter metrics to observability platform

### 5.2 CI Benchmark Gate
- Every pull request touching the execution path must run the CI benchmark suite
- CI must produce benchmark artifacts for: replay workload AND peak-load synthetic workload
- Pull request is **blocked** if p99 regresses beyond the degrade threshold

### 5.3 Degrade Safeguard Behaviour
When Fast Loop p99 > 10 ms:
1. Emit `FASTLOOP_DEGRADE` alert
2. Switch to bypass mode: use last validated policy snapshot; skip in-process compute stages
3. Continue order management; do not halt trading unless risk limits breached
4. Log all degrade events with duration and cause
5. Restore normal mode only after 3 consecutive p99 measurements below 8 ms

### 5.4 Weekly Latency Report
- Owner reviews p50/p99/p99.9 trend weekly
- Any regression triggers remediation ticket with assigned owner and resolution deadline
- Report retained for 6 months

---

## 6. SAFE RL TRAINING & DEPLOYMENT

### 6.1 Promotion Gate (No Exceptions)
- Online learning gated behind offline training AND paper trading validation
- Exploration constrained by risk limits and hard safety filters during all training
- Promotion to live requires risk committee approval AND tested rollback readiness
- Online RL micro-updates: **DEFERRED to Tier 3**; do not implement in current cycle

### 6.2 Deliberation Process
- Heavy reasoning (simulations, policy search) stays in Slow Loop; never execution path
- Trade rejection triggers: risk-control breach, stale-data TTL breach, latency-cap breach
- Trade rejection is **never** triggered by pending simulation completion

---

## 7. OBSERVATION SPACE SCHEMA (versioned)
```
{
  "technical": {price_features, volume_features, microstructure_features},
  "regime": {regime_label, regime_probability, ood_flag, alien_state_flag},
  "sentiment": {z_t, freshness_flag, confidence},
  "macro": {var_es_levels, macro_differentials, rbi_stance},
  "consensus": {signal, confidence, mode},
  "order_book": {imbalance, queue_pressure, quality_flag},
  "schema_version": "string"
}
```
- Schema version incremented on any field addition, removal, or type change
- Downstream consumers validate schema version on every read

---

## EXECUTION AGENT CHECKLIST
- [ ] Student policy artifact: deterministic runtime verified, compression plan approved
- [ ] Teacher-student crisis-slice agreement threshold passed
- [ ] Fast Loop p99 <= 8 ms on replay and peak-load benchmarks
- [ ] Degrade path tested: p99 > 10 ms triggers bypass correctly
- [ ] Order-book imbalance: feature schema versioned, quality guardrails active
- [ ] Impact monitor: threshold buckets defined, auto-reduction tested in paper trading
- [ ] CI benchmark gate active on execution-path pull requests
- [ ] Capacity limits hardcoded and validated in pre-trade checks
- [ ] Online RL micro-updates NOT enabled (Tier 3 deferred)
- [ ] Weekly latency report scheduled with named owner
