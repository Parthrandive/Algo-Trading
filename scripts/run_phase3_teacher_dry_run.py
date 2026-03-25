from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.strategic.action_space import export_week2_action_space
from src.agents.strategic.artifacts import Phase3RunManifest, resolve_code_hash, utc_now, write_manifest
from src.agents.strategic.config import (
    DEFAULT_ACTION_EXPORT_DIR,
    DEFAULT_ACTION_EXPORT_FILE,
    DEFAULT_RUN_MANIFEST_FILE,
    STRATEGIC_EXEC_CONTRACT_VERSION,
    WEEK2_ACTION_EXPORT_VERSION,
)
from src.agents.strategic.observation import ObservationAssembler
from src.agents.strategic.placeholder_teacher import generate_placeholder_teacher_actions
from src.agents.strategic.registry import register_placeholder_teacher_policies
from src.db.phase3_recorder import Phase3Recorder


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_symbols(value: str) -> list[str]:
    symbols = [item.strip() for item in value.split(",") if item.strip()]
    deduped: list[str] = []
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        deduped.append(symbol)
        seen.add(symbol)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DB-backed Week 1 dry-run for placeholder teacher actions (Phase 3 strategic foundation)."
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. RELIANCE.NS,TCS.NS).")
    parser.add_argument("--start", required=True, help="ISO8601 start timestamp (UTC recommended).")
    parser.add_argument("--end", required=True, help="ISO8601 end timestamp (UTC recommended).")
    parser.add_argument("--database-url", default=None, help="Optional SQLAlchemy database URL override.")
    parser.add_argument("--policy-id", default="teacher_placeholder_v0", help="Placeholder teacher policy id.")
    parser.add_argument("--batch-size", type=int, default=500, help="Observation materialization batch size.")
    parser.add_argument("--output-dir", default=str(DEFAULT_ACTION_EXPORT_DIR), help="Artifact output directory.")
    parser.add_argument(
        "--action-export-file",
        default=DEFAULT_ACTION_EXPORT_FILE,
        help="Week 2 action-space export filename (JSONL).",
    )
    parser.add_argument(
        "--manifest-file",
        default=DEFAULT_RUN_MANIFEST_FILE,
        help="Run manifest filename.",
    )
    parser.add_argument(
        "--no-materialize-observations",
        action="store_true",
        help="Read from Phase 2 DB and build actions, but do not write observations table.",
    )
    parser.add_argument(
        "--write-decisions",
        action="store_true",
        help="Persist placeholder teacher actions to trade_decisions.",
    )
    parser.add_argument(
        "--register-teacher-model-cards",
        action="store_true",
        help="Optionally wire placeholder SAC/PPO/TD3 teachers into existing model registry tables.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise ValueError("No symbols provided.")
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    if end < start:
        raise ValueError("end must be greater than or equal to start.")

    run_started = utc_now()
    run_id = f"phase3_week1_dry_run_{run_started.strftime('%Y%m%dT%H%M%SZ')}"
    output_root = Path(args.output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    assembler = ObservationAssembler(database_url=args.database_url)
    recorder = Phase3Recorder(database_url=args.database_url)

    # Build observations directly from real Phase 2 DB tables.
    observations = []
    per_symbol_counts: dict[str, int] = {}
    warn_count = 0
    for symbol in symbols:
        rows = assembler.build_symbol_observations(symbol=symbol, start=start, end=end)
        per_symbol_counts[symbol] = len(rows)
        warn_count += sum(1 for row in rows if row.quality_status != "pass")
        observations.extend(rows)
        if not args.no_materialize_observations and rows:
            for idx in range(0, len(rows), max(1, args.batch_size)):
                chunk = rows[idx : idx + max(1, args.batch_size)]
                recorder.save_observation_batch([item.model_dump(mode="python") for item in chunk])

    actions = generate_placeholder_teacher_actions(observations, policy_id=args.policy_id)

    rewards = []
    for action in actions:
        action_name = action.action.value
        directional_bonus = 1.0 if action_name in {"buy", "sell"} else 0.0
        reward_value = float(action.confidence) * directional_bonus - (0.01 * float(action.action_size))
        rewards.append(
            {
                "timestamp": action.timestamp,
                "symbol": action.symbol,
                "policy_id": action.policy_id,
                "reward_name": "ra_drl_composite_placeholder",
                "reward_value": reward_value,
                "components": {
                    "confidence": float(action.confidence),
                    "directional_bonus": directional_bonus,
                    "size_penalty": 0.01 * float(action.action_size),
                    "placeholder": True,
                },
            }
        )

    if args.write_decisions and actions:
        payload = []
        for action in actions:
            row = action.model_dump(mode="json")
            row["is_placeholder"] = True
            payload.append(row)
        recorder.save_trade_decision_batch(payload)

    if rewards:
        recorder.save_reward_log_batch(rewards)

    export_path = export_week2_action_space(output_root / args.action_export_file, actions)

    policy_ids = []
    training_runs_written = 0
    if args.register_teacher_model_cards:
        policy_ids = register_placeholder_teacher_policies(
            database_url=args.database_url,
            run_id=run_id,
            artifact_root=output_root,
            export_path=export_path,
            register_model_cards=True,
        )
        for policy_id in policy_ids:
            elapsed = max(0.0, (utc_now() - run_started).total_seconds())
            recorder.save_rl_training_run(
                {
                    "policy_id": policy_id,
                    "run_timestamp": run_started,
                    "training_start": start,
                    "training_end": end,
                    "episodes": max(1, len(observations)),
                    "total_steps": max(1, len(observations)),
                    "final_reward": 0.0,
                    "dataset_snapshot_id": run_id,
                    "code_hash": resolve_code_hash(),
                    "duration_seconds": elapsed,
                    "notes": "placeholder week1 dry-run training registration",
                }
            )
            training_runs_written += 1

    run_finished = utc_now()
    manifest = Phase3RunManifest(
        run_id=run_id,
        started_at_utc=run_started,
        finished_at_utc=run_finished,
        symbols=symbols,
        observation_schema_version=observations[0].observation_schema_version if observations else "1.0",
        contract_version=STRATEGIC_EXEC_CONTRACT_VERSION,
        export_schema_version=WEEK2_ACTION_EXPORT_VERSION,
        rows_materialized=0 if args.no_materialize_observations else len(observations),
        actions_generated=len(actions),
        code_hash=resolve_code_hash(),
        dataset_snapshot={
            "phase2_window": {"start": start.isoformat(), "end": end.isoformat()},
            "per_symbol_observation_rows": per_symbol_counts,
            "warn_quality_rows": warn_count,
        },
        artifacts={
            "action_export_jsonl": str(export_path),
            "write_decisions": bool(args.write_decisions),
            "materialized_observations": not args.no_materialize_observations,
            "reward_logs_written": len(rewards),
            "training_runs_written": training_runs_written,
            "registered_policy_ids": policy_ids,
        },
        notes=[
            "Teacher actions are placeholder-only and restricted to offline/slow-loop usage.",
            "Fast Loop exclusion for teacher inference remains enforced by contract schema.",
        ],
    )
    manifest_path = write_manifest(output_root / args.manifest_file, manifest)

    summary = {
        "run_id": run_id,
        "symbols": symbols,
        "rows_built": len(observations),
        "rows_materialized": 0 if args.no_materialize_observations else len(observations),
        "actions_generated": len(actions),
        "reward_logs_written": len(rewards),
        "training_runs_written": training_runs_written,
        "action_export": str(export_path),
        "manifest": str(manifest_path),
        "registered_policy_ids": policy_ids,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
