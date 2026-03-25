from __future__ import annotations

from src.agents.strategic.config import PolicyFoundationConfig
from src.agents.strategic.policies.base import TeacherPolicyFoundation


class TD3PolicyFoundation(TeacherPolicyFoundation):
    def __init__(self) -> None:
        super().__init__(
            config=PolicyFoundationConfig(policy_id="phase3_td3_teacher_v1", algorithm="TD3"),
            dependency_name="stable_baselines3",
            default_hyperparameters={
                "learning_rate": 1e-3,
                "buffer_size": 100_000,
                "batch_size": 256,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
            },
        )
