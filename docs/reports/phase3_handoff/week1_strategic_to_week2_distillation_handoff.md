# Week 1 Strategic to Week 2 Distillation Handoff

Date: March 25, 2026  
Scope: Phase 3 Week 1 -> Week 2 handoff

## 1) What Is Delivered
- Real DB-backed observation materialization from Phase 2 outputs.
- Finalized strategic -> strategic_executive contract (`strategic_exec_v1`).
- Locked Week 2 action-space export format (`week2_action_space_v1`).
- Standard run manifest (`phase3_run_manifest_v1`) for future checkpoints.
- Optional model-registry wiring for placeholder teacher policies.

## 2) Data Flow (Week 1 Final)
1. Read Phase 2 tables (`technical_predictions`, `regime_predictions`, `sentiment_scores`, `consensus_signals`).
2. Materialize aligned rows into `observations`.
3. Generate placeholder teacher actions (slow-loop only).
4. Export action-space JSONL for Week 2 consumption.
5. Emit `run_manifest.json` with code hash, window, row counts, artifact paths.

## 3) Action-Space Export Format
Per-line JSON record fields:
- `export_schema_version`
- `contract_version`
- `timestamp`
- `symbol`
- `policy_id`
- `policy_type`
- `loop_type`
- `action`
- `action_size`
- `confidence`
- `observation_id`
- `snapshot_id`
- `observation_schema_version`
- `quality_status`
- `decision_reason`

## 4) Command for DB-Backed Dry Run
```bash
python scripts/run_phase3_teacher_dry_run.py \
  --symbols RELIANCE.NS,TCS.NS \
  --start 2026-03-01T00:00:00Z \
  --end 2026-03-25T00:00:00Z
```

Optional:
- `--write-decisions`
- `--register-teacher-model-cards`
- `--no-materialize-observations`

## 5) Week 2 Inputs
- `observations` table with schema version `1.0`
- `week2_action_space_export.jsonl`
- `run_manifest.json`
- optional `rl_policies` + `model_cards` placeholder entries

## 6) Safety Constraints Carried Forward
- Teacher policies remain excluded from Fast Loop.
- Fast Loop candidate path in Week 2 is student-policy-only after distillation and latency gating.
