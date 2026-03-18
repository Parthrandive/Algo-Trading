# Textual Data Agent

The Textual Data Agent collects and processes textual data from approved sources for sentiment analysis and risk assessment in the algo-trading system.

## Overview

This agent ingests text from:
- NSE News (RSS feeds)
- Economic Times (news feeds)
- NewsAPI market/finance feed (official API)
- RBI Reports (official RSS/XML and DBIE downloads)
- Earnings Transcripts (PDF extraction)
- X Posts (social media with keyword filters)

It produces canonical records validated against schemas (`NewsArticle_v1.0`, `SocialPost_v1.0`, `EarningsTranscript_v1.0`) and sidecar metadata for operational diagnostics.

## Inputs

- **Runtime Config**: `configs/textual_data_agent_runtime_v1.json` - Contains source allowlists, compliance rules, X query templates, and runtime flags.
- **As-of Timestamp**: Optional UTC datetime for historical runs (defaults to current time).
- **Database Recorder**: Optional `SilverDBRecorder` for persistence (defaults to in-memory export).

## Outputs

- **Canonical Records**: Schema-validated text records persisted to Silver DB.
- **Sidecar Metadata**: Operational diagnostics in `logs/textual_sidecar_records.json`, including confidence, TTL, manipulation risk, and compliance status.
- **Compliance Logs**: Rejected records logged in `logs/compliance_rejects.log`.
- **PDF Spot Checks**: Quality metrics in `logs/textual_pdf_spot_check_report.json`.

## Schema

### Canonical Payloads
All records include:
- `source_id`: Unique identifier
- `timestamp`: Event time (UTC)
- `content`: Normalized text
- `source_type`: e.g., "rss_feed", "social_post"
- `language`: "en", "hi", "code_mixed"
- `ingestion_timestamp_utc/ist`: Processing timestamps
- `schema_version`: "1.0"
- `quality_status`: "pass", "warn", "fail"

Type-specific fields as per schemas.

### Sidecar Metadata
- `source_route_detail`: "primary_api", "official_feed", "fallback_scraper"
- `quality_flags`: List of issues
- `manipulation_risk_score`: 0.0-1.0
- `confidence`: 0.0-1.0
- `ttl_seconds`: Freshness TTL
- `compliance_status`: "allow", "reject"
- `compliance_reason`: Optional string

## Usage

### Running the Agent

```python
from src.agents.textual.textual_data_agent import TextualDataAgent

agent = TextualDataAgent.from_default_components()
batch = agent.run_once()
print(f"Processed {len(batch.canonical_records)} records")
```

### With Custom Config

```python
from pathlib import Path
agent = TextualDataAgent.from_default_components(runtime_config_path=Path("custom_config.json"))
```

### With Database Persistence

```python
from src.db.silver_db_recorder import SilverDBRecorder
recorder = SilverDBRecorder()
agent = TextualDataAgent.from_default_components(recorder=recorder)
batch = agent.run_once()
# Records are persisted automatically
```

## Failure Handling

- **Source Outages**: Adapters handle retries and fallbacks (e.g., RBI scraper only in emergencies).
- **Compliance Rejections**: Logged with reasons; records not persisted.
- **Schema Validation Failures**: Records marked as "fail" in sidecar; not persisted to canonical DB.
- **PDF Extraction Errors**: Quality score < 0.6 marks as "fail"; logged in spot checks.
- **Rate Limits**: X adapter implements exponential backoff.
- **Deduplication**: Fingerprint-based; duplicates logged as rejections.

## Dependencies

- `pydantic` for schemas
- `pdfplumber` for PDF extraction
- `urllib` for HTTP requests
- Standard library for XML/HTML parsing

## Testing

Run tests with:
```bash
pytest tests/test_textual_*.py
```

Gate evidence available in `logs/textual_day6_*` files.

## Configuration

Edit `configs/textual_data_agent_runtime_v1.json` for:
- Source URLs and keywords
- Compliance rules
- Emergency flags (e.g., RBI fallback)

Set `NEWS_API_KEY` in your environment (or `.env`) to enable the NewsAPI market/finance adapter.

## Monitoring

Check logs for:
- Compliance rejects
- PDF quality metrics
- Sidecar diagnostics

For production, integrate with observability dashboard.
