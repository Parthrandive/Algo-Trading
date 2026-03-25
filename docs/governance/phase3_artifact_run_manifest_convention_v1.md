# Phase 3 Artifact & Run Manifest Convention (v1)

Date: March 25, 2026  
Applies to: Strategic Week 1 and later checkpoints

## 1) Required Artifacts Per Run
Each Phase 3 strategic run must emit:
- `run_manifest.json`
- `week2_action_space_export.jsonl`
- optional policy/model metadata files under the same run directory

Recommended root:
- `data/reports/phase3/week1/<run_id>/`

## 2) Manifest Schema
Run manifest version is `phase3_run_manifest_v1` and includes:
- `run_id`
- `started_at_utc`
- `finished_at_utc`
- `symbols`
- `observation_schema_version`
- `contract_version`
- `export_schema_version`
- `rows_materialized`
- `actions_generated`
- `code_hash`
- `dataset_snapshot`
- `artifacts`
- `notes`

The schema is implemented in `src/agents/strategic/artifacts.py` (`Phase3RunManifest`).

## 3) Naming Rules
- `run_id` format: `phase3_week1_dry_run_YYYYMMDDTHHMMSSZ`
- Export file: `week2_action_space_export.jsonl`
- Manifest file: `run_manifest.json`

## 4) Reproducibility Requirements
- `code_hash` must capture the active git revision when available.
- `dataset_snapshot` must include the DB window (`start`, `end`) and per-symbol row counts.
- `symbols` in manifest must match symbols used for materialization/action generation.

## 5) Forward Compatibility
- Any breaking change in manifest fields requires a manifest version bump.
- Week 2+ components must read version fields before parsing payload content.
