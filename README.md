# Multi-Agent AI Trading System
### Integrated Plan for Indian Markets — USD/INR · NSE Stocks & Indices

> **Version 1.2.1 · February 2026**  
> Status: Pre-production · Target: Fall 2026 deployment

---

## Overview

A production-grade multi-agent AI trading system purpose-built for Indian markets, covering NSE equities, indices, USD/INR forex, and F&O instruments. The architecture combines real-time market data ingestion, FinBERT-based sentiment analysis, RL-based trade execution, and independent risk oversight into a dual-loop system deployed on a Mumbai-based low-latency VPS.

**Key design goals:**
- ≥20% improvement in event-driven accuracy via sentiment integration
- Sub-10 ms Fast Loop (execution-critical path)
- SEBI & broker-compliant audit trail
- Layered kill-switch and crisis-mode safety hierarchy

---

## Table of Contents

- [Architecture](#architecture)
- [System Phases](#system-phases)
  - [Phase 1 — Data Orchestration](#phase-1--data-orchestration)
  - [Phase 2 — Analyst Board (Modeling)](#phase-2--analyst-board-modeling)
  - [Phase 3 — Strategic Executive (Trader Ensemble)](#phase-3--strategic-executive-trader-ensemble)
  - [Phase 4 — Independent Risk Overseer](#phase-4--independent-risk-overseer)
  - [Phase 5 — Validation & Evolution Loop](#phase-5--validation--evolution-loop)
- [Asset Universe & Constraints](#asset-universe--constraints)
- [Regulatory & Compliance](#regulatory--compliance)
- [Risk & Portfolio Management](#risk--portfolio-management)
- [Stress Testing](#stress-testing)
- [MLOps & Model Governance](#mlops--model-governance)
- [Rollout Strategy](#rollout-strategy)
- [Go-Live Checklist](#go-live-checklist)
- [Performance Metrics](#performance-metrics)

---

## Architecture

The system enforces a strict **dual-loop separation**:

| Loop | Latency Target | Responsibilities |
|------|---------------|-----------------|
| **Fast Loop** | p99 ≤ 10 ms | Technical microstructure, order-imbalance features, cached sentiment reads (O(1) Redis) |
| **Slow Loop** | 100–500 ms typical | Heavy inference, regime updates, sentiment fine-tuning, policy refresh — fully async, never blocks order placement |

**Infrastructure:** Mumbai-based low-latency VPS with static IP whitelisting. Volatility bypass thresholds use hysteresis and cooldown windows to prevent mode flapping.

```
NSE Sentinel ──┐
Macro Monitor ──┤──▶ Preprocessing ──▶ Analyst Board ──▶ Trader Ensemble ──▶ Execution
Textual Data ──┘                         (FinBERT / RL)        (SAC/PPO/TD3)
                                                  │
                                       Independent Risk Overseer
                                       (Kill Switch · SHAP · ADDM)
```

---

## System Phases

### Phase 1 — Data Orchestration

Four specialized agents handle all data acquisition and preprocessing.

#### NSE Sentinel Agent
- Sources: `nsepython`, `jugad-data`, official NSE APIs (web scraping restricted to fallback)
- Redundant feeds for price, volume, and corporate actions
- Timestamp validation and clock sync enforced
- Primary feed failure → automatic **reduce-only / close-only** mode

#### Macro Monitor Agent
- Indicators: WPI, CPI, IIP, FII/DII flows, FX reserves, RBI bulletins, India–US 10Y spread
- Global proxies (Brent, DXY) explicitly justified for India relevance
- Data quality checks for missingness, outliers, and latency

#### Preprocessing Agent
- Deterministic, reproducible feature engineering
- Rolling-window normalization and directional change thresholds
- Corporate action validation against a reference source
- Leakage tests with strict time alignment and feature lag checks
- Feature promotion pipeline: proposal → offline eval → shadow test → production

#### Textual Data Agent
- Sources: NSE news, Economic Times, RBI reports, earnings transcripts, X (keyword + semantic search)
- Code-mixed English / Hinglish handling strategy documented
- PDF extraction pipeline with spot-check validation

---

### Phase 2 — Analyst Board (Modeling)

#### Technical Agent
- Models: ARIMA-LSTM hybrids, 2D CNNs, GARCH quantiles (VaR & ES)
- Baselines and ablation tests required for all model families

#### Regime Agent
- Model: PEARL or equivalent meta-learning
- Regime definitions include RBI inner/outer band zones
- Structural break stress validation: 2008 GFC, 2013 Taper Tantrum, COVID window
- **Alien-state detection** via explicit statistical distance thresholds → staged de-risking (full risk → reduced risk → neutral/cash)

#### Sentiment Agent
- Base model: `ProsusAI/finbert` (or equivalent with documented justification)
- Fine-tuning: `harixn/indian_news_sentiment`, SEntFiN
- **Dual-speed design:**
  - *Intraday fast lane*: lightweight model + keyword rules, target ≤100 ms from headline arrival
  - *Offline slow lane*: full FinBERT retraining, weekly cadence
- Outputs written to Redis cache with timestamp, confidence score, and TTL
- Cache policy: fresh → use · stale → downweight · expired → ignore
- Cache read failure → technical-only reduced-risk execution mode
- Pump-and-dump / slang-scam detection for code-mixed text

#### Consensus Agent
- Aggregation: LSTAR / ESTAR with Bayesian estimation
- Transition function includes volatility, macro differentials, RBI signals, and sentiment quantiles
- **Basic crisis mode**: crisis-weighted routing, max 60–70% dominance cap for crisis agent
- Agent divergence treated as a dedicated regime signal
- Full winner-takes-all automation deferred until shadow-mode stability thresholds are met

---

### Phase 3 — Strategic Executive (Trader Ensemble)

#### Ensemble Composition
`SAC` · `PPO` · `TD3` (or justified alternatives), each with independent training cadence and evaluation metrics.

#### Decision Mechanism
- Maximum entropy framework
- Multi-threshold genetic algorithm for action selection (offline only)
- **Teacher-student distillation** for live execution: student policy mimics teacher under strict latency constraints
- Student promotion requires above-threshold agreement on **crisis slices**, not just average-day performance

#### Observation Space
Regime probabilities · VaR/ES levels · macro differentials · sentiment scores · technical features — versioned and schema-validated.

#### Reward & Utility Functions
Reward library: RA-DRL composite · Sharpe · Sortino · Calmar · Omega · Kelly variants  
Dynamic weighting tied to regime state and sentiment. Downside penalties increase during RBI outer band breach or negative sentiment spikes.

#### Deliberation Process
- Critical-path thinker: lightweight and deterministic (stays in Fast Loop)
- Policy snapshot refreshed every ~10 minutes or earlier on trigger
- Trade rejection tied to: risk-control breach · stale-data TTL breach · latency-cap breach
- Hard cap: Fast Loop p99 ≤ 10 ms

#### Safe RL Training
- Online learning gated behind offline training and paper trading validation
- Exploration constrained by risk limits and hard safety filters
- Live promotion requires risk committee approval and rollback readiness

---

### Phase 4 — Independent Risk Overseer

| Control | Detail |
|---------|--------|
| Kill switch | Layered hierarchy: model → portfolio → broker → manual |
| Loss limits | Max drawdown, daily loss, position size — enforced at broker and system level |
| Drift detection | ADDM or equivalent |
| Explainability | SHAP logging, ≥80% of trades with top-k feature explanations |
| Operating modes | normal · reduce-only · close-only · kill-switch |

**Crisis taxonomy:**

| State | Description | Response |
|-------|-------------|----------|
| Full crisis | Volatility break + liquidity deterioration + agent confidence threshold breach | Staged de-risking |
| Agent divergence | Fundamental disagreement between agents | Neutral-hold with staged re-risking on alignment recovery |
| Slow-crash / freeze | Gradual deterioration or feed freeze | Reduce-only mode, safe-mode entry |

Crisis entry requires **multi-condition confirmation** persisting for a configured tick/time window. Crisis mode has max-duration expiry and auto-reverts unless revalidated.

---

### Phase 5 — Validation & Evolution Loop

- **Backtesting**: Vectorbt or equivalent, spanning 2018/2019–present (COVID crash, budget cycles, elections, Oct–Nov 2022 volatility window)
- Walk-forward evaluation and time-based cross-validation required
- Survivorship bias controls with point-in-time universe data (including delisted symbols)
- Nightly incremental learning (TTM or equivalent lightweight models)
- PEARL meta-learning adapts to most recent 1–3 month regimes
- Weekly sentiment re-fine-tuning validated against SEntFiN
- Dual-loop boundary tests verify text floods alone cannot flip portfolio direction without confirming technical/regime signals

---

## Asset Universe & Constraints

**Universe filters (NSE equities, indices, USD/INR, F&O):**

| Filter | Threshold |
|--------|-----------|
| Avg Daily Turnover | ≥ ₹10 Cr (6-month average) |
| Impact Cost | < NSE IWF methodology threshold |
| Price | ≥ ₹50 |
| Listing Age | ≥ 12 months |
| Circuit Hits | ≤ 5 in last 3 months |

- Primary trading frequency: **Hourly** (daily timeframe for confirmation/regime context only)
- Universe reviewed and rebalanced **quarterly** by the Preprocessing Agent
- Capacity: Initial live notional ₹5–10 Cr · Scale to ₹25 Cr only after 6 months of live validation · Hard cap at 2x if impact exceeds 0.25% at 3x

---

## Regulatory & Compliance

- SEBI and broker compliance checklist maintained
- Pre-trade checks: position limits, margin rules, circuit breaker rules
- Audit trail: order intent, execution, cancellation, and model version per trade
- Static Mumbai IP whitelisting (production only)
- Regulatory reporting automation and retention policy documented
- Algorithm approval/notification obligations documented
- **Data usage policy strictly prohibits unpublished or embargoed information**

---

## Risk & Portfolio Management

- Risk budgets defined per asset class and sector
- Correlation and concentration limits enforced
- Position sizing tied to forecast uncertainty and liquidity
- Max participation limits enforced at execution time
- Downside risk emphasis enforced during panic regimes

---

## Stress Testing

- Maintained scenario library with versioned definitions and expected outcomes
- Explicit shock scenarios: RBI surprise rate hike · INR flash move · liquidity drought
- Historical scenarios: 2008 GFC · 2013 Taper Tantrum
- Synthetic shocks at 1x, 2x, and 3x capacity
- **Impossible-scenario tests**: correlation inversion, frozen constituent prices, simultaneous feed failures
- Data poisoning and feed-freeze simulations
- Snapback tests measuring exposure clipping speed after flash-shock conditions
- Results reviewed quarterly → trigger risk budget or capacity adjustments

---

## MLOps & Model Governance

- Model registry includes data snapshot and code hash
- Champion-challenger promotion gates codified
- Rollback policy exists and is tested
- Training pipelines reproducible from raw data
- Human oversight gates with defined review cadence and approval authority
- Material strategy changes (latency bypass, dual-speed sentiment, novelty safeguards) require ablation evidence before deployment

---

## Rollout Strategy

1. **Shadow mode** — minimum fixed window, full monitoring, no live capital
2. **Promotion gate** — challenger must outperform champion on risk-adjusted metrics with no regression in Sharpe, drawdown, or slippage
3. **Sequential override rollout** — enable one override at a time, validate, then promote the next
4. **Live A/B** — optional, gated by risk committee approval
5. Monitoring dashboards: PnL attribution · drift · risk limits · mode-switch frequency · OOD trigger rate · kill-switch activations

---

## Go-Live Checklist

| Metric | Minimum Threshold |
|--------|------------------|
| Paper trading period | ≥ 3 calendar months, ≥80% uptime |
| Annualized Sharpe | ≥ 1.8 |
| Sortino | ≥ 2.0 |
| Max Drawdown | ≤ 8% |
| Win Rate | ≥ 52% |
| Profit Factor | ≥ 1.5 |
| Avg Realized Slippage | ≤ model estimate + 20 bps |
| Data Uptime (NSE hours) | ≥ 99.5% |
| Compliance | Zero critical violations · zero broker/SEBI flags |

---

## Performance Metrics

Core metrics tracked across all stages:

`Sharpe` · `Sortino` · `Calmar` · `Max Drawdown` · `Turnover` · `Slippage` · `Capacity Utilization` · `Fill Rate` · `Mode-Switch Frequency` · `OOD Trigger Rate`

Promotion gates include statistical significance thresholds vs. baseline and risk budget adherence validation.

---

## Document Control

| Field | Value |
|-------|-------|
| Version | 1.2.1 |
| Date | February 2026 |
| Key changes in v1.2 | Dual-loop execution, sentiment cache decoupling, crisis override voting |
| Key changes in v1.2.1 | Minimum-safe scope: ship dual-loop, sentiment cache, basic crisis mode, impossible-state tests · defer snapback smoothing tuning and full winner-takes-all automation |

---

> **Disclaimer:** Performance targets are stated as goals, not guarantees, with confidence intervals. This system is subject to SEBI regulations and all applicable Indian financial market laws.
