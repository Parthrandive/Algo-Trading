# Week 2 Regime Agent to Week 3 Sentiment Agent Handoff

Date: March 15, 2026
Scope: Phase 2 Week 2 Day 7 handoff

## 1) Regime Agent Output Contract
The Regime Agent publishes the following fields for downstream consumers:
- `regime_state`: Bull, Bear, Sideways, Crisis, RBI-Band transition, Alien
- `transition_probability`: likelihood of changing regime (0.0 to 1.0)
- `confidence`: ensemble confidence (0.0 to 1.0)
- `risk_level`: `full_risk`, `reduced_risk`, or `neutral_cash`
- `model_id`: current ensemble identifier

## 2) Interpretation Guidance for Sentiment Agent
- Use `regime_state` to select sentiment weighting mode:
  - Bull/Sideways: standard sentiment weighting
  - Bear/Crisis: down-weight pro-risk sentiment
  - Alien: only high-confidence defensive sentiment signals
- Use `risk_level` as a hard risk gate before converting sentiment to trade intent.

## 3) Interface Expectations for Consensus Agent
- Consensus Agent should combine:
  - Technical direction/confidence
  - Regime risk gating (`risk_level`)
  - Sentiment directional confidence
- If `risk_level=neutral_cash`, consensus must not emit pro-risk execution.

## 4) Operational Notes
- Daily scheduler can run Regime inference after preprocessing.
- Model cards are available under:
  - `data/models/hmm_regime/model_card.json`
  - `data/models/pearl_meta/model_card.json`
  - `data/models/ood_detector/model_card.json`

## 5) Known Gaps
- HMM/PEARL baselines are deterministic approximations and should be replaced with full production training loops in Week 3 hardening.
