# Module Boundary Notes v1
Date: 2026-02-10
Scope: Week 1 Day 2 architecture lock for Phase 1 data orchestration.

## 1. Boundary Rules
- Every module reads from upstream contracts and writes contract-validated outputs only.
- Every output record must include provenance metadata and quality status.
- Fast-loop consumers never invoke connector/network calls directly.
- Failure in one module degrades downstream mode, but does not bypass schema validation.

## 2. Module I/O Contracts
| Module | Inputs | Outputs | Failure Behavior |
| :--- | :--- | :--- | :--- |
| `NSE Sentinel Agent` | Official exchange/broker feeds, fallback scraper feed, source health events | Raw feed events (Bronze), canonical market records (Silver), feed health state | On primary feed loss, switch to fallback, mark `source_type=fallback_scraper`, emit degrade advisory (`reduce-only` / `close-only`) |
| `Macro Monitor Agent` | RBI, Commerce, AMFI/SEBI, FX/macro releases | Canonical macro indicator events (Silver) with freshness metadata | On source outage, emit last-known-good with stale marker and quality downgrade |
| `Preprocessing Agent` | Silver market/macro tables and corp action events | Gold feature tables and deterministic transform logs | On schema mismatch, reject batch and quarantine symbol scope |
| `Textual Data Agent` | News feeds, filings, transcripts, social streams | Canonical text metadata + content payloads for sentiment pipeline | On extraction/parse failure, quarantine record with `quality_status=fail` |
| `Schema Registry` | Version key + payload | Validated typed model instance or validation error | On unknown schema version, fail closed and block downstream publish |

## 3. Interface IDs
- `market.tick.v1`
- `market.bar.v1`
- `market.corporate_action.v1`
- `macro.indicator.v1`
- `text.news_article.v1`
- `text.social_post.v1`
- `text.earnings_transcript.v1`

## 4. State Transition Contract
- Allowed operating states: `normal`, `reduce-only`, `close-only advisory`.
- State changes are triggered by module health events, never by ad hoc manual data edits.
- Recovery requires health signal stabilization and schema-valid fresh data publication.
