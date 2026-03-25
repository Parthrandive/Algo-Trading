from __future__ import annotations

from src.agents.strategic.config import PolicyFoundationConfig
from src.agents.strategic.policies.base import TeacherPolicyFoundation


class SACPolicyFoundation(TeacherPolicyFoundation):
    def __init__(self) -> None:
        super().__init__(
            config=PolicyFoundationConfig(policy_id="phase3_sac_teacher_v1", algorithm="SAC"),
            dependency_name="stable_baselines3",
            default_hyperparameters={
                "learning_rate": 3e-4,
                "buffer_size": 100_000,
                "batch_size": 256,
                "tau": 0.005,
            },
        )
