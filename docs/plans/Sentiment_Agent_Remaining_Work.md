# Sentiment Agent — Remaining Work Plan

**Status**: Week 3 PARTIAL | **Date**: March 21, 2026  
**Ref**: Phase_2_Analyst_Board_Plan.md → Week 3 (March 23–29)

---

## What's Already Built ✅

### Data Ingestion Pipeline (fully operational)
- `textual_data_agent.py` — orchestrator with `run_once()` method
- `adapters.py` — 10+ data adapters:
  - `NSECorporateAnnouncementsAdapter` (NSE filings)
  - `EconomicTimesMarketsAdapter` (ET RSS feed)
  - `NewsApiMarketAdapter` (NewsAPI.org)
  - `RBIReportsAdapter` (RBI bulletins & press releases)
  - `XPostsAdapter` (Twitter/X social posts)
  - `EarningsTranscriptsAdapter` (earnings calls)
  - File-level caching per adapter (`data/textual_cache/`)
- `cleaners.py` — text cleaning, HTML stripping, normalisation
- `validators.py` — TTL-based freshness, quality scoring, duplicate detection
- `exporters.py` — DB export via `SilverDBRecorder`

### Safety & Language Services
- `safety_service.py` — pump-and-dump detection, spam filtering, adversarial pattern detection (caps, punctuation bursts, link spam)
- `language_service.py` — Hinglish detection, code-mixed text handling
- `pdf_service.py` — PDF text extraction for RBI reports

### Database
- `text_items` table: **141 records** (139 news, 1 social, 1 transcript)
- `sentiment_scores` table: **117 records** scored by pretrained FinBERT
- `transformers` v5.3.0 installed

---

## What's Missing ❌

### 1. FinBERT Fine-Tuning on Indian Market Data
**Priority**: HIGH | **Effort**: 1 day

**What**: The current setup uses pretrained ProsusAI/finbert without fine-tuning. Indian market language (RBI policy, SEBI, Nifty context) gets misclassified.

**Tasks**:
- [ ] Download `harixn/indian_news_sentiment` dataset from HuggingFace
- [ ] Download `SEntFiN` (Financial Entity Sentiment) dataset
- [ ] Write fine-tuning script: `scripts/finetune_finbert.py`
  - Input: HF dataset → tokenize → train/val split
  - Training: 3-5 epochs, lr=2e-5, weight decay=0.01
  - Eval: precision/recall/F1 per class (positive/negative/neutral)
- [ ] Save fine-tuned model to `data/models/sentiment/finbert_indian_v1/`
- [ ] Update sentiment scoring to use fine-tuned model instead of base

**Pass criteria**: Precision ≥ 0.75, Recall ≥ 0.70 on Indian market test set

**Files to create**:
- `scripts/finetune_finbert.py`
- `data/models/sentiment/finbert_indian_v1/` (output dir)

---

### 2. Fast Lane — Intraday Sentiment Scorer
**Priority**: HIGH | **Effort**: 1 day

**What**: Lightweight keyword-rule scorer for real-time headline scoring. Must respond in ≤ 100ms (no GPU, no transformer inference).

**Tasks**:
- [ ] Create `src/agents/sentiment/fast_lane.py`
  - Keyword dictionaries: bullish/bearish Indian market terms
  - RBI policy keywords (rate cut → positive, tightening → negative)
  - SEBI action keywords (ban, investigation → negative)
  - Pattern: headline → keyword match → score + confidence
- [ ] Target latency: ≤ 100ms per headline
- [ ] Output format: `{sentiment_class, sentiment_score, confidence, lane='fast'}`
- [ ] Write to `sentiment_scores` with `lane='fast'`

**Pass criteria**: Latency ≤ 100ms, agreement with FinBERT ≥ 60% on test set

**Files to create**:
- `src/agents/sentiment/fast_lane.py`
- `tests/test_fast_lane.py`

---

### 3. Slow Lane — Nightly Deep Sentiment Pipeline
**Priority**: HIGH | **Effort**: 1 day

**What**: Nightly batch pipeline that runs fine-tuned FinBERT on all text items collected during the day.

**Tasks**:
- [ ] Create `src/agents/sentiment/slow_lane.py`
  - Fetch all `text_items` from last 24h
  - Score each with fine-tuned FinBERT
  - Write to `sentiment_scores` with `lane='slow'`
- [ ] Compute daily aggregate sentiment variable `z_t`:
  - `z_t = weighted_mean(sentiment_scores, weights=confidence)`
  - Separate z_t per symbol + one MARKET-level z_t
  - Store z_t in `sentiment_scores` with `lane='daily_agg'`
- [ ] Integrate with scheduler (cron or APScheduler)

**Pass criteria**: All items from last 24h scored, z_t computed and stored

**Files to create**:
- `src/agents/sentiment/slow_lane.py`
- `scripts/run_nightly_sentiment.py` (entry point)

---

### 4. Sentiment Cache Policy
**Priority**: MEDIUM | **Effort**: 0.5 day

**What**: Deterministic cache policy for how stale sentiment scores are handled downstream.

**Tasks**:
- [ ] Create `src/agents/sentiment/cache_policy.py`
  - `FRESH` (age < TTL): use full weight
  - `STALE` (TTL < age < 2×TTL): downweight by 50%
  - `EXPIRED` (age > 2×TTL): ignore completely
- [ ] Define TTLs per source type:
  - News headlines: TTL = 4 hours
  - Social media: TTL = 1 hour
  - Earnings transcripts: TTL = 7 days
  - RBI reports: TTL = 30 days
- [ ] Implement cache-failure fallback:
  - If ALL sentiment is expired → switch to technical-only reduced-risk mode
  - Log cache miss events for monitoring
- [ ] Integrate with ConsensusAgent (read cache status before applying sentiment weight)

**Pass criteria**: Stale scores are auto-downweighted, expired scores ignored

**Files to create**:
- `src/agents/sentiment/cache_policy.py`

---

### 5. Unified Sentiment Agent Class
**Priority**: MEDIUM | **Effort**: 0.5 day

**What**: A single `SentimentAgent` class that owns both lanes and exposes a clean API for the Consensus Agent.

**Tasks**:
- [ ] Create `src/agents/sentiment/sentiment_agent.py`
  - `score_realtime(headline) → SentimentScore` (fast lane)
  - `run_nightly_batch() → list[SentimentScore]` (slow lane)
  - `get_z_t(symbol) → float` (aggregated daily sentiment)
  - `get_cached_sentiment(symbol) → SentimentScore` (with cache policy)
- [ ] Wire Phase2Recorder for all writes
- [ ] Register model card in DB for sentiment agent

**Files to create**:
- `src/agents/sentiment/__init__.py`
- `src/agents/sentiment/sentiment_agent.py`

---

### 6. Precision/Recall Evaluation Report
**Priority**: MEDIUM | **Effort**: 0.5 day

**What**: Classification report proving sentiment meets thresholds.

**Tasks**:
- [ ] Create `scripts/eval_sentiment.py`
  - Load Indian test set
  - Run fine-tuned FinBERT
  - Output: precision, recall, F1 per class
  - Output: confusion matrix
- [ ] Verify thresholds: precision ≥ 0.75, recall ≥ 0.70
- [ ] Save report to `data/reports/sentiment_eval/classification_report.json`

**Files to create**:
- `scripts/eval_sentiment.py`

---

### 7. Sentiment Stability Tests
**Priority**: LOW | **Effort**: 0.5 day

**What**: Tests ensuring sentiment can't single-handedly flip portfolio direction.

**Tasks**:
- [ ] Create `tests/test_sentiment_stability.py`
  - Test: flood of 100 negative headlines doesn't override LONG signal when technical + regime agree
  - Test: single extreme sentiment score can't exceed max weight cap
  - Test: expired cache triggers reduced-risk fallback
  - Test: Hinglish text gets valid scores (not NaN or crash)

**Files to create**:
- `tests/test_sentiment_stability.py`

---

## Summary Table

| # | Task | Priority | Effort | Status |
|---|---|---|---|---|
| 1 | FinBERT fine-tuning on Indian data | HIGH | 1 day | ❌ |
| 2 | Fast lane (keyword scorer) | HIGH | 1 day | ❌ |
| 3 | Slow lane (nightly FinBERT) | HIGH | 1 day | ❌ |
| 4 | Cache policy | MEDIUM | 0.5 day | ❌ |
| 5 | Unified SentimentAgent class | MEDIUM | 0.5 day | ❌ |
| 6 | Precision/recall eval report | MEDIUM | 0.5 day | ❌ |
| 7 | Stability tests | LOW | 0.5 day | ❌ |

**Total estimated effort**: ~5 days

---

## Recommended Build Order

```
Day 1: FinBERT fine-tuning (#1) — everything else depends on this
Day 2: Fast lane (#2) + Slow lane (#3) — the two scoring engines
Day 3: Unified SentimentAgent (#5) + Cache policy (#4) — wire everything
Day 4: Eval report (#6) + Stability tests (#7) — prove it works
```
