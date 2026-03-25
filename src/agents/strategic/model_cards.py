from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.agents.strategic.schemas import RLPolicyRegistryEntry, RLTrainingRunRecord


def build_teacher_model_card(
    policy: RLPolicyRegistryEntry,
    run: RLTrainingRunRecord | None = None,
    *,
    limitations: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "model_id": policy.policy_id,
        "agent": "strategic",
        "model_family": policy.algorithm,
        "version": policy.version,
        "created_at": policy.created_at,
        "updated_at": now,
        "status": "foundation_only",
        "teacher_policy": True,
        "offline_only": policy.offline_only,
        "training_status": policy.training_status,
        "observation_schema_version": policy.observation_schema_version,
        "action_space": policy.action_space,
        "checkpoint_status": policy.checkpoint_status,
        "checkpoint_path": policy.checkpoint_path,
        "hyperparameters": dict(policy.metadata.get("hyperparameters", {})),
        "latest_run": run.model_dump() if run is not None else None,
        "known_limitations": limitations
        or [
            "No Week 1 training has been executed yet.",
            "Teacher policies must remain offline-only and out of the Fast Loop.",
        ],
        "promotion_gate": {
            "training_complete": False,
            "fast_loop_ready": False,
            "distillation_required": True,
        },
        "metadata": dict(extra_metadata or {}),
    }
