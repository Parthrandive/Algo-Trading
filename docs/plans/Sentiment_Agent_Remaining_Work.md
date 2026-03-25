# Sentiment Agent - Completion Update

**Status**: Engineering scope implemented on March 21, 2026  
**Plan Ref**: Phase 2 Analyst Board -> Week 3 sentiment buildout

## What Was Already Present
- Text ingestion and cleaning via `src/agents/textual/`
- Safety and Hinglish handling services
- Phase 2 sentiment primitives in `src/agents/sentiment/`
- Phase 2 DB recorder support for basic sentiment rows

## What Was Completed

### 1. Dual-speed sentiment runtime
- Added explicit fast lane wrapper: `src/agents/sentiment/fast_lane.py`
- Added explicit slow lane wrapper: `src/agents/sentiment/slow_lane.py`
- Extended the unified runtime: `src/agents/sentiment/sentiment_agent.py`

### 2. Deterministic cache policy
- Added source-aware TTL policy in `src/agents/sentiment/cache_policy.py`
- Implemented source buckets and TTLs:
  - News: 4 hours
  - Social: 1 hour
  - Earnings transcripts: 7 days
  - RBI reports: 30 days
- Added downstream freshness evaluation for `fresh`, `stale`, and `expired`

### 3. Unified SentimentAgent API
- `score_realtime(...)`
- `run_nightly_batch(...)`
- `get_z_t(symbol)`
- `get_cached_sentiment(symbol)`
- DB-backed model-card registration

### 4. DB persistence and metadata
- Expanded `sentiment_scores` persistence to store:
  - `source_id`
  - `source_type`
  - `ttl_seconds`
  - `freshness_flag`
  - `headline_timestamp`
  - `score_timestamp`
  - `quality_status`
  - `metadata_json`
- Updated recorder and query helpers accordingly

### 5. Fine-tuning and evaluation workflow
- Added offline-safe training helpers: `src/agents/sentiment/training.py`
- Added training script: `scripts/finetune_finbert.py`
- Added evaluation script: `scripts/eval_sentiment.py`
- Added nightly entrypoint: `scripts/run_nightly_sentiment.py`

### 6. Local artifacts generated
- Model artifact bootstrap dir: `data/models/sentiment/finbert_indian_v1/`
- Bootstrap model card: `data/models/sentiment/finbert_indian_v1/model_card.json`
- Bootstrap training metadata: `data/models/sentiment/finbert_indian_v1/training_meta.json`
- Evaluation report: `data/reports/sentiment_eval/classification_report.json`

## Workflow Now

### Real-time path
1. Headline enters `SentimentAgent.score_realtime(...)`
2. Fast lane scores it with keyword rules and safety filters
3. Source-aware TTL is attached
4. Score is optionally persisted to `sentiment_scores`
5. Consensus or downstream consumers call `get_cached_sentiment(symbol)`
6. Cache policy returns full weight, downweighted score, or reduced-risk fallback

### Nightly path
1. `scripts/run_nightly_sentiment.py` loads the last 24 hours of `text_items`
2. Slow lane scores each document
3. Per-symbol and market-wide `z_t` aggregates are computed
4. Slow-lane document scores and `daily_agg` rows are persisted
5. `get_z_t(symbol)` exposes the latest aggregate for downstream agents

### Training/eval path
1. `scripts/finetune_finbert.py` trains the offline-safe sentiment artifact
2. Artifact files are written under `data/models/sentiment/finbert_indian_v1/`
3. `scripts/eval_sentiment.py` produces the classification report JSON
4. `register_model_card()` stores the sentiment model card in Phase 2 registry storage

## Test Coverage Added
- `tests/test_fast_lane.py`
- `tests/test_sentiment_stability.py`
- `tests/test_sentiment_workflows.py`

## Verification
- Expanded sentiment suite passed on March 21, 2026:
  - `tests/test_sentiment_day1.py`
  - `tests/test_sentiment_day3_day6.py`
  - `tests/test_phase2_db.py`
  - `tests/test_fast_lane.py`
  - `tests/test_sentiment_stability.py`
  - `tests/test_sentiment_workflows.py`
- Result: `24 passed`

## Honest Remaining Limitation
- The local bootstrap artifact is intentionally marked `bootstrap_only`.
- It uses a synthetic offline corpus because the machine currently lacks the full Hugging Face `transformers` / `datasets` path and the real Indian labeled datasets were not fetched in this turn.
- The engineering workflow is complete, but promotion against the plan thresholds still requires:
  - real `harixn/indian_news_sentiment` data
  - real `SEntFiN` data
  - a real fine-tuning run
  - a threshold-passing evaluation report on non-synthetic data
