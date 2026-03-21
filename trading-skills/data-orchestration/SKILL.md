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

## 6. WORKED EXAMPLES

These procedures were validated on real tasks. Follow the exact steps when performing similar work.

### 6.1 Backfilling a Macro Indicator from FRED

**Validated:** March 2026 — backfilled CPI, WPI, IIP, FX_RESERVES, INDIA_US_10Y_SPREAD.

**Problem:** The `macro_indicators` table had < 10 rows for most indicators. Training models treated macro features as all-NaN.

**FRED Series Reference (public CSV, no API key):**
| Indicator | FRED Series ID | Frequency |
|---|---|---|
| CPI | `INDCPIALLMINMEI` | Monthly |
| WPI | `WPIATT01INM661N` | Monthly |
| IIP | `INDPRINTO01IXOBM` | Monthly |
| FX_RESERVES | `TRESEGINM052N` | Monthly |
| INDIA_US_10Y_SPREAD | `INDIRLTLT01STM` + `DGS10` | Monthly (computed) |

**Steps:**
1. **Dry-run first** — always preview before writing to DB:
   ```bash
   python scripts/backfill_all_macro.py --dry-run
   ```
2. **Backfill all** (writes to `macro_indicators` via `SilverDBRecorder.save_macro_indicators()`):
   ```bash
   python scripts/backfill_all_macro.py
   ```
3. **Backfill specific indicators:**
   ```bash
   python scripts/backfill_all_macro.py --indicators CPI WPI --start 2015-01-01
   ```
4. **Verify** — check DB row counts:
   ```sql
   SELECT indicator_name, COUNT(*), MIN(timestamp), MAX(timestamp)
   FROM macro_indicators GROUP BY indicator_name ORDER BY indicator_name;
   ```
5. **Validate coverage** — run the macro validation script:
   ```bash
   python scripts/validate_macro_backfill.py --symbol RELIANCE.NS --interval 1h
   ```
   Expected: `coverage_pct >= 60%` and `train_ready = True` for all backfilled indicators.

**Gotcha — Python SSL vs curl:** On Anaconda Python, `urllib` and `requests` may time out on FRED due to SSL library differences. The backfill script uses `curl` subprocess as a workaround. If curl also fails, download CSVs manually from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID`.

### 6.2 Adding a NEW Macro Indicator End-to-End

**Steps:**
1. Add the enum value to `MacroIndicatorType` in `src/schemas/macro_data.py`
2. Find the FRED series ID at https://fred.stlouisfed.org/ (search for the indicator)
3. Add the series definition to `FRED_SERIES` dict in `scripts/backfill_all_macro.py`
4. Add the column name to `MACRO_COLUMNS` list in `src/agents/technical/data_loader.py`
5. Add the z-score feature name to `src/agents/technical/features.py`
6. Run: `python scripts/backfill_all_macro.py --indicators NEW_INDICATOR --dry-run`
7. If dry-run succeeds: `python scripts/backfill_all_macro.py --indicators NEW_INDICATOR`
8. Validate: `python scripts/validate_macro_backfill.py --symbol RELIANCE.NS`
9. Retrain models to pick up the new feature

---

## 7. PERFORMANCE BASELINE (Skill vs No-Skill)

Measured on: **March 2026 — Macro data backfill task**

### Without this skill
```
Task: "Backfill macro indicators into the database"
- Agent tries MOSPI scraper first → 30s timeout
- Agent tries Python urllib to FRED → 30s SSL timeout
- Agent tries Python requests to FRED → 15s SSL timeout
- Agent manually guesses FRED series IDs (gets some wrong)
- Agent doesn't know SilverDBRecorder API → reads 3 files to figure it out
- Agent doesn't validate after insert → silent data gaps

Result:
  Attempts before success: 6+
  Time wasted on SSL debugging: ~10 minutes
  Records inserted: 0 (gave up or inserted wrong format)
  Validation: none performed
```

### With this skill (§6.1 procedure)
```
Task: "Backfill macro indicators into the database"
- Agent reads §6.1 → knows exact FRED series IDs
- Agent reads Gotcha → uses curl (skips Python SSL issue)
- Agent runs: python scripts/backfill_all_macro.py --dry-run
- Agent verifies: 1,354 records fetched
- Agent runs: python scripts/backfill_all_macro.py
- Agent validates: DB query confirms 8,443 total rows

Result:
  Commands run: 2 (dry-run + real run)
  Time to completion: ~5 seconds
  Records inserted: 1,354
  Validation: DB row counts + date ranges confirmed
```

### Success criteria
| Metric | Without Skill | With Skill | Target |
|---|---|---|---|
| Commands to complete task | 6+ retry attempts | 2 commands | ≤ 3 |
| Time to first successful insert | >10 min (may never succeed) | ~5 seconds | < 1 minute |
| Records correctly inserted | 0 | 1,354 | > 0 |
| Post-insert validation | None | DB query + date ranges | Always |
| Known gotchas avoided | 0 | 1 (Python SSL) | All documented |

---

## AGENT CHECKLIST (run before any downstream handoff)
- [ ] All feeds have provenance tags
- [ ] No feed is in degraded state without ops alert
- [ ] All features pass leakage test (100 %)
- [ ] Universe filters applied and version logged
- [ ] Sentiment cache TTL status checked; policy applied
- [ ] Macro indicators complete; no STALE flags without logged justification
- [ ] Replay artifact determinism test passes
