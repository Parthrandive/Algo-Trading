# Week 4 Textual Data Agent Day 1 Design Note
Date: 2026-03-02
Status: Initial implementation committed

## Scope Delivered (Day 1)
- Source allowlist and legal/compliance policy config for textual feeds.
- Canonical schema mapping freeze to existing registry keys:
  - `NewsArticle_v1.0`
  - `SocialPost_v1.0`
  - `EarningsTranscript_v1.0`
- Sidecar metadata schema for sentiment/risk handoff (`TextSidecarMetadata_v1.0`).
- Textual agent skeleton with modular boundaries:
  - `adapters.py`
  - `cleaners.py`
  - `validators.py`
  - `exporters.py`
  - `textual_data_agent.py`
- Baseline India-market X query templates (keyword + semantic intents).

## Implemented Artifacts
- Runtime policy and query templates:
  - `configs/textual_data_agent_runtime_v1.json`
- Sidecar schema:
  - `src/schemas/text_sidecar.py`
- Agent skeleton:
  - `src/agents/textual/`

## Workflow (Current Skeleton)
1. Adapter pull:
   - A source adapter returns `RawTextRecord` objects tagged with `source_name`, `source_type`, and `source_route_detail`.
2. Cleaner normalize:
   - `TextCleaner` strips noisy whitespace and standardizes content text.
3. Canonical mapping:
   - `TextualDataAgent` maps each cleaned record into canonical payload fields based on `record_type`.
4. Compliance gate:
   - `TextualValidator.evaluate_compliance` checks:
     - source allowlist membership
     - allowed provenance route
     - publish/embargo/license checks
     - required public timestamp and source URL
5. Canonical validation:
   - Allowed records are validated via `SchemaRegistry` against frozen schema IDs.
6. Sidecar build:
   - `TextSidecarMetadata` is generated for every processed record, including compliance status, confidence, TTL, and risk diagnostics.
7. Export split:
   - `TextualExporter` returns separate canonical and sidecar collections so canonical payloads remain strict (`extra=forbid`) and DB-safe.

## Next Day Dependencies
- Day 2 can now plug real collectors into adapter stubs.
- Persistence wiring can use existing `SilverDBRecorder.save_text_items` for canonical and a new sidecar sink.
- CP1 freeze evidence can reference the runtime config and schema mapping in this note.
