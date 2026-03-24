from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agents.strategic.config import PolicyFoundationConfig
from src.agents.strategic.schemas import RLPolicyRegistryEntry, RLTrainingRunRecord, StrategicObservation


@dataclass
class TeacherPolicyFoundation:
    config: PolicyFoundationConfig
    dependency_name: str | None = None
    default_hyperparameters: dict[str, Any] = field(default_factory=dict)

    @property
    def policy_name(self) -> str:
        return self.config.policy_id

    def registry_entry(self) -> RLPolicyRegistryEntry:
        checkpoint_path = self.config.checkpoint_root / self.config.policy_id / "checkpoint.pending"
        return RLPolicyRegistryEntry(
            policy_id=self.config.policy_id,
            algorithm=self.config.algorithm,
            action_space=self.config.action_space,
            observation_schema_version=self.config.observation_schema_version,
            checkpoint_path=str(checkpoint_path),
            checkpoint_status="pending_training",
            notes="Foundation scaffold created. Teacher remains offline-only until trained.",
            metadata={
                "dependency": self.dependency_name,
                "hyperparameters": self.default_hyperparameters,
                "training_enabled": False,
            },
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    def planned_run(self) -> RLTrainingRunRecord:
        return RLTrainingRunRecord(
            policy_id=self.config.policy_id,
            started_at=datetime.now(UTC),
            status="planned",
            reward_name="step_return",
            metrics={"status": "foundation_only"},
            params={"hyperparameters": self.default_hyperparameters, "training_enabled": False},
            checkpoint_path=str(self.config.checkpoint_root / self.config.policy_id / "checkpoint.pending"),
            notes="No training executed. Placeholder run record created for Week 1 foundation.",
        )

    def placeholder_action(self, observation: StrategicObservation) -> tuple[float, float]:
        signal = float(observation.observation_vector[13]) * 0.6 + float(observation.observation_vector[11]) * 0.4
        confidence = max(0.0, min(1.0, float(observation.observation_vector[14])))
        return max(-1.0, min(1.0, signal)), confidence

    def ensure_dependency(self) -> None:
        if self.dependency_name is None:
            return
        __import__(self.dependency_name)

    def training_entrypoint(self) -> Path:
        return self.config.checkpoint_root / self.config.policy_id
