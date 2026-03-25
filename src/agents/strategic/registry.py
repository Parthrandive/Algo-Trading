from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.agents.strategic.config import OBSERVATION_SCHEMA_VERSION
from src.db.phase2_recorder import Phase2Recorder
from src.db.phase3_recorder import Phase3Recorder


def register_placeholder_teacher_policies(
    *,
    database_url: str | None,
    run_id: str,
    artifact_root: Path,
    export_path: Path,
    register_model_cards: bool = True,
) -> list[str]:
    """
    Optionally wire Week 1 placeholder teacher artifacts into existing model registry.
    """
    now = datetime.now(timezone.utc)
    policy_ids = [
        f"sac_teacher_{run_id}",
        f"ppo_teacher_{run_id}",
        f"td3_teacher_{run_id}",
    ]
    phase3 = Phase3Recorder(database_url=database_url)
    phase2 = Phase2Recorder(database_url=database_url)

    for algorithm, policy_id in zip(("SAC", "PPO", "TD3"), policy_ids):
        phase3.save_rl_policy(
            {
                "policy_id": policy_id,
                "algorithm": algorithm,
                "version": "0.1.0",
                "status": "candidate",
                "created_at": now,
                "artifact_path": str(artifact_root),
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "reward_function": "ra_drl_composite",
                "hyperparams": {
                    "status": "placeholder",
                    "notes": "week 1 placeholder registration without training",
                },
                "training_metrics": {},
                "compression_method": "none",
            }
        )
        if register_model_cards:
            phase2.save_model_card(
                {
                    "model_id": policy_id,
                    "agent": "strategic",
                    "model_family": "rl_teacher",
                    "version": "0.1.0",
                    "status": "candidate",
                    "created_at": now,
                    "updated_at": now,
                    "performance": {},
                    "metadata": {
                        "run_id": run_id,
                        "algorithm": algorithm,
                        "artifact_root": str(artifact_root),
                        "action_export_path": str(export_path),
                        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                        "placeholder": True,
                    },
                },
                model_id=policy_id,
                agent="strategic",
            )
    return policy_ids
