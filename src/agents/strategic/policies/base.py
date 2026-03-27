from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.agents.strategic.config import PolicyFoundationConfig
from src.agents.strategic.environment import StrategicTradingEnv
from src.agents.strategic.reward import CompositeRewardConfig, trading_performance_summary
from src.agents.strategic.schemas import RLPolicyRegistryEntry, RLTrainingRunRecord, StrategicObservation


@dataclass
class PolicyTrainResult:
    policy_id: str
    algorithm: str
    checkpoint_path: Path
    episodes: int
    total_steps: int
    final_reward: float
    duration_seconds: float
    metrics: dict[str, Any]
    params: dict[str, Any]
    train_start: datetime
    train_end: datetime
    reward_name: str
    notes: str | None = None

    def as_policy_row(self, *, observation_schema_version: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "policy_id": self.policy_id,
            "algorithm": self.algorithm,
            "version": "1.0.0",
            "status": "candidate",
            "created_at": now,
            "artifact_path": str(self.checkpoint_path),
            "observation_schema_version": observation_schema_version,
            "reward_function": self.reward_name,
            "hyperparams": self.params,
            "training_metrics": self.metrics,
            "compression_method": "none",
        }

    def as_training_run_row(
        self,
        *,
        dataset_snapshot_id: str | None = None,
        code_hash: str | None = None,
    ) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "started_at": self.train_start,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "final_reward": self.final_reward,
            "sharpe": self.metrics.get("sharpe"),
            "sortino": self.metrics.get("sortino"),
            "max_drawdown": self.metrics.get("max_drawdown"),
            "win_rate": self.metrics.get("win_rate"),
            "dataset_snapshot_id": dataset_snapshot_id,
            "code_hash": code_hash,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.capacity = int(max(1, capacity))
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._ptr
        self.state[idx] = np.asarray(state, dtype=np.float32)
        self.action[idx] = np.asarray(action, dtype=np.float32)
        self.reward[idx, 0] = float(reward)
        self.next_state[idx] = np.asarray(next_state, dtype=np.float32)
        self.done[idx, 0] = 1.0 if done else 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if self._size == 0:
            raise ValueError("ReplayBuffer is empty")
        idx = np.random.randint(0, self._size, size=int(batch_size))
        state = torch.as_tensor(self.state[idx], dtype=torch.float32, device=device)
        action = torch.as_tensor(self.action[idx], dtype=torch.float32, device=device)
        reward = torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=device)
        done = torch.as_tensor(self.done[idx], dtype=torch.float32, device=device)
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return self._size


@dataclass
class TeacherPolicyFoundation:
    config: PolicyFoundationConfig
    dependency_name: str | None = "torch"
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
            notes="Offline teacher policy scaffold. Training runs must stay outside Fast Loop.",
            metadata={
                "dependency": self.dependency_name,
                "hyperparameters": self.default_hyperparameters,
                "training_enabled": True,
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
            metrics={"status": "planned"},
            params={"hyperparameters": self.default_hyperparameters, "training_enabled": True},
            checkpoint_path=str(self.config.checkpoint_root / self.config.policy_id / "checkpoint.pending"),
            notes="Offline training planned. Teacher remains restricted to slow/offline loop.",
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

    @staticmethod
    def resolve_device(device: str | None = None) -> torch.device:
        requested = (device or "cpu").strip().lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")

    @staticmethod
    def seed_everything(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_env(
        self,
        *,
        observations: list[StrategicObservation],
        prices: list[float],
        reward_name: str,
        reward_config: CompositeRewardConfig | None = None,
    ) -> StrategicTradingEnv:
        return StrategicTradingEnv(
            observations=observations,
            prices=prices,
            reward_name=reward_name,
            reward_config=reward_config or CompositeRewardConfig(),
            action_space_kind=self.config.action_space,
        )

    def evaluate_policy(
        self,
        *,
        observations: list[StrategicObservation],
        prices: list[float],
        reward_name: str,
        action_fn,
        reward_config: CompositeRewardConfig | None = None,
    ) -> dict[str, Any]:
        env = self.build_env(
            observations=observations,
            prices=prices,
            reward_name=reward_name,
            reward_config=reward_config,
        )
        obs, _ = env.reset()
        episode_reward = 0.0
        step_net_returns: list[float] = []
        done = False
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = action_fn(obs_tensor)
            action_value = float(torch.as_tensor(action).detach().cpu().reshape(-1)[0].item())
            outcome = env.step(np.asarray([action_value], dtype=np.float32))
            if len(outcome) == 5:
                obs, reward, done, _, info = outcome
            else:
                obs, reward, done, info = outcome
            episode_reward += float(reward)
            step_info = info.get("step_result", {})
            step_net_returns.append(float(step_info.get("net_return", reward)))

        summary = trading_performance_summary(step_net_returns)
        summary["episode_reward"] = float(episode_reward)
        summary["final_portfolio_value"] = float(env.reward_logs[-1].portfolio_value) if env.reward_logs else 0.0
        summary["reward_logs"] = [item.model_dump(mode="json") for item in env.reward_logs]
        summary["step_net_returns"] = step_net_returns
        return summary

    @staticmethod
    def save_checkpoint(payload: dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def train_offline(
        self,
        *,
        observations: list[StrategicObservation],
        prices: list[float],
        total_timesteps: int,
        seed: int,
        reward_name: str,
        output_dir: Path,
        device: str | None = None,
        reward_config: CompositeRewardConfig | None = None,
        **kwargs: Any,
    ) -> PolicyTrainResult:
        raise NotImplementedError
