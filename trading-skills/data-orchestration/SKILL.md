# SKILL: Data Orchestration Layer
**Project:** Multi-Agent AI Trading System — Indian Market (NSE / USD-INR / MCX Gold)
**Applies To:** NSE Sentinel Agent · Macro Monitor Agent · Preprocessing Agent · Textual Data Agent
**Version:** aligned with Plan v1.3.7

---

## PURPOSE
This skill governs every agent that acquires, validates, normalises, and stores raw market data. No downstream model or execution component may consume data that has not passed all checks defined here.

---

## 1. NSE SENTINEL AGENT

### 1.1 Data Sources (Priority Order)
1. Official NSE API or broker-provided exchange feed (PRIMARY)
2. `jugad-data` or `nsepython` as secondary/fallback
3. Web scraping — FALLBACK ONLY; never primary; tag every record with `source=scraper`

### 1.2 Mandatory Feed Checks
- **Timestamp validation**: every tick must carry exchange-stamped UTC timestamp; reject records with drift > 200 ms versus NTP reference
- **Clock sync**: enforce NTP sync; drift alert fires if system clock diverges > 100 ms
- **Provenance tag**: every event carries `source` field (`primary_api` | `fallback_scraper` | `replay`)
- **Completeness**: core symbol completeness >= 99.0 % during NSE trading hours; alert on breach

### 1.3 Failure Modes & Escalation
| Condition | Action |
|-----------|--------|
| Primary feed down | Switch to secondary; log `FEED_FAILOVER` event |
| Secondary feed down | Enter `reduce-only` mode; block new position opens |
| Both feeds down | Enter `close-only` mode; alert operator immediately |
| Integrity check fails | Block order placement; do not clear block until integrity check passes |
| Feed restored | Run integrity checks before exiting degraded mode |

### 1.4 Data Quality SLAs
- Feed uptime >= 99.5 % during NSE hours (09:15–15:30 IST) measured over 10 consecutive trading days
- Max permissible ingest-to-feature latency: p99 <= defined budget per Section 4 of main plan
- Anomaly alert (outlier price, zero-volume freeze, tick gap) triggers automatic fallback or pause

---

## 2. MACRO MONITOR AGENT

### 2.1 Required Indicators (versioned list)
| Indicator | Source | Frequency |
|-----------|--------|-----------|
| WPI | MoSPI / RBI | Monthly |
| CPI | MoSPI | Monthly |
| IIP | MoSPI | Monthly |
| FII / DII flows | NSE / SEBI | Daily |
| FX Reserves | RBI Weekly Statistical Supplement | Weekly |
| RBI MPC Bulletins | RBI website | Event-driven |
| India–US 10Y Spread | Bloomberg / CCIL / manual | Daily |
| Brent Crude | ICE / CME | Daily (justified: India import cost proxy) |
| DXY | ICE | Daily (justified: USD/INR directional pressure) |

### 2.2 Quality Checks
- **Missingness**: if any Tier-1 macro indicator is missing, log warning and forward-fill for up to 3 periods; beyond 3 periods, mark as `STALE` and reduce regime confidence
- **Outlier detection**: z-score > 4 on any indicator triggers `MACRO_ANOMALY` flag; do not silently accept
- **Latency audit**: each indicator carries `data_as_of` and `received_at` timestamps; alert if lag exceeds defined SLA

### 2.3 Agent Outputs
- Versioned macro feature vector with `as_of_date`, `source_list`, and `quality_flags`
- Regime-relevant macro differential: India–US 10Y spread, RBI stance tag (accommodative / neutral / hawkish)

---

## 3. PREPROCESSING AGENT

### 3.1 Normalisation Rules
- Rolling window normalisation only; window size is locked per feature family at design time
- Directional change threshold is documented and version-controlled
- All normalisation parameters are fitted on training data only; never on val/test window

### 3.2 Corporate Action Adjustments
- Price and volume series must be adjusted for splits, bonuses, rights, and dividends before use
- Adjusted series are validated against a reference source (e.g., NSE Bhav Copy)
- Each adjustment event is logged with date, type, factor, and reference source

### 3.3 Leakage Prevention (MANDATORY)
- Every feature must be strictly time-lagged; no look-ahead permitted
- Run leakage tests on every feature change:
  - Timestamp audit: feature `t` uses only data with `as_of <= t`
  - Correlation test: future-return correlation on shuffled labels must not exceed baseline
- Leakage test pass rate must be 100 % before feature promotion

### 3.4 Universe Selection & Rebalance
Apply numeric filters quarterly:
- Avg Daily Turnover >= ₹10 Cr (6-month average)
- Impact Cost <= 0.50 %
- Free Float >= 20 % (NSE IWF methodology)
- Price >= ₹50
- Listing age >= 12 months
- Exclude: any stock with > 5 circuit hits in last 3 months

Universe change log must be versioned. Rebalance only via this agent.

### 3.5 Feature Approval Process
1. **Propose**: feature spec with definition, lag, source, and expected signal rationale
2. **Offline evaluation**: correlation, importance, leakage tests
3. **Shadow testing**: run in shadow alongside live features for >= 2 weeks
4. **Promotion**: require sign-off; update feature registry with hash and version

### 3.6 Data Tiering Policy
| Tier | Storage | Retention | Access Pattern |
|------|---------|-----------|---------------|
| Hot | In-memory / Redis cache | Current session + 48 h | O(1) read; Fast Loop |
| Warm | Low-latency DB (e.g., TimescaleDB) | 12 months | Feature server queries |
| Cold | Object store / archive | 7 years (compliance) | Replay / audit only |

---

## 4. TEXTUAL DATA AGENT

### 4.1 Sources
- NSE official announcements and corporate filings
- Economic Times, Mint, LiveMint financial news
- RBI press releases, MPC minutes, bulletins
- Earnings call transcripts (BSE / company IR)
- X (Twitter): keyword + semantic search; documented keyword list required

### 4.2 Text Quality & Robustness
- **Deduplication**: hash-based; duplicate articles must not inflate sentiment scores
- **Spam / adversarial filtering**: reject records flagged as pump-and-dump, slang-scam, or hype burst; log rejection with reason
- **Code-mixed handling**: Hinglish / English-Hindi text must be handled via documented strategy (transliteration layer or multilingual model)
- **PDF extraction**: validated with quarterly spot checks; extraction errors logged

### 4.3 Sentiment Pipeline (Dual-Speed)
| Lane | Model | Latency Target | Update Cadence |
|------|-------|----------------|----------------|
| Fast (intraday) | Lightweight model + keyword rules | <= 100 ms from headline arrival | Near real-time |
| Slow (deep) | FinBERT fine-tuned on Indian data | Seconds to minutes | Nightly batch |

- Fast lane writes to Redis cache with: `score`, `confidence`, `timestamp`, `TTL`
- **Cache decision policy** (deterministic):
  - `fresh` (within TTL) → use score
  - `stale` (TTL exceeded but < 2× TTL) → downweight by 50 %
  - `expired` (> 2× TTL) → ignore; execution falls back to technical-only reduced-risk mode
- Cache read failure → technical-only reduced-risk execution; alert ops

### 4.4 Manipulation Detection
- Pump-and-dump patterns or slang-scam bursts → force sentiment downgrade to `neutral`
- Sentiment-to-price mismatch circuit breaker → if sentiment strongly positive but price rapidly declining (or vice versa), downgrade to `neutral` and log `SENTIMENT_MISMATCH`

---

## 5. DATA SCALE & REPLAY

### 5.1 Replay Framework Requirements
- Must reconstruct any trading day from raw feed payloads → feature artifacts
- All intermediate artifacts carry deterministic identifiers (hash + version)
- Supports both **event-time** and **wall-clock** playback modes
- Replay output must be byte-identical across runs (determinism test on CI)

### 5.2 Quarterly Data Roadmap Review
- Track current vs target daily processing volume (GB → TB)
- Report replay coverage gaps and data quality regressions
- Update tiering policy if volumes breach tier thresholds

---

## AGENT CHECKLIST (run before any downstream handoff)
- [ ] All feeds have provenance tags
- [ ] No feed is in degraded state without ops alert
- [ ] All features pass leakage test (100 %)
- [ ] Universe filters applied and version logged
- [ ] Sentiment cache TTL status checked; policy applied
- [ ] Macro indicators complete; no STALE flags without logged justification
- [ ] Replay artifact determinism test passes
