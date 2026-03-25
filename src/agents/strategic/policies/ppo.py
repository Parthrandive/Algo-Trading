from __future__ import annotations

from src.agents.strategic.config import PolicyFoundationConfig
from src.agents.strategic.policies.base import TeacherPolicyFoundation


class PPOPolicyFoundation(TeacherPolicyFoundation):
    def __init__(self) -> None:
        super().__init__(
            config=PolicyFoundationConfig(policy_id="phase3_ppo_teacher_v1", algorithm="PPO"),
            dependency_name="stable_baselines3",
            default_hyperparameters={
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "clip_range": 0.2,
                "gae_lambda": 0.95,
            },
        )
