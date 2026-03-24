from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from src.agents.strategic.config import EnvironmentCostConfig
from src.agents.strategic.reward import step_return_reward
from src.agents.strategic.schemas import RewardLog, StepResult, StrategicObservation

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None


class _FallbackEnv:
    """Small stand-in so foundation tests work without gymnasium installed."""


@dataclass
class StrategicTradingEnv(_FallbackEnv if gym is None else gym.Env):
    observations: list[StrategicObservation]
    prices: list[float]
    reward_name: str = "step_return"
    cost_config: EnvironmentCostConfig = field(default_factory=EnvironmentCostConfig)
    action_space_kind: str = "continuous"

    def __post_init__(self) -> None:
        if len(self.observations) < 2:
            raise ValueError("StrategicTradingEnv requires at least two observations")
        if len(self.prices) != len(self.observations):
            raise ValueError("prices and observations must have the same length")
        self._episode_id = ""
        self._idx = 0
        self._position = 0.0
        self._portfolio_value = float(self.cost_config.initial_cash)
        self.reward_logs: list[RewardLog] = []
        if gym is not None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.observations[0].observation_vector),),
                dtype=np.float32,
            )
            if self.action_space_kind == "discrete":
                self.action_space = gym.spaces.Discrete(3)
            else:
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._idx = 0
        self._position = 0.0
        self._portfolio_value = float(self.cost_config.initial_cash)
        self._episode_id = f"strategic-{uuid4().hex[:12]}"
        self.reward_logs = []
        obs = np.asarray(self.observations[self._idx].observation_vector, dtype=np.float32)
        return obs, {"episode_id": self._episode_id, "offline_only": True}

    def step(self, action: float | int | np.ndarray):
        if self._idx >= len(self.observations) - 1:
            raise RuntimeError("Environment already completed. Call reset() before stepping again.")

        target_position = self._normalize_action(action)
        current_price = float(self.prices[self._idx])
        next_price = float(self.prices[self._idx + 1])
        price_return = (next_price / current_price) - 1.0 if current_price else 0.0
        position_before = float(self._position)
        turnover = abs(target_position - position_before)
        transaction_cost = turnover * (self.cost_config.brokerage_bps / 10_000.0)
        slippage_cost = turnover * (
            (self.cost_config.slippage_bps + self.cost_config.impact_bps_per_unit * abs(target_position)) / 10_000.0
        )
        gross_return = target_position * price_return
        reward = step_return_reward(gross_return, transaction_cost, slippage_cost)
        self._portfolio_value *= 1.0 + reward
        self._position = target_position
        self._idx += 1
        done = self._idx >= len(self.observations) - 1

        result = StepResult(
            reward=reward,
            gross_return=gross_return,
            net_return=reward,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            position_before=position_before,
            position_after=self._position,
            portfolio_value=self._portfolio_value,
            done=done,
            metadata={"price_return": price_return},
        )
        self.reward_logs.append(
            RewardLog(
                symbol=self.observations[self._idx - 1].symbol,
                timestamp=self.observations[self._idx - 1].timestamp,
                episode_id=self._episode_id,
                reward_name=self.reward_name,
                reward_value=result.reward,
                portfolio_value=result.portfolio_value,
                gross_return=result.gross_return,
                net_return=result.net_return,
                transaction_cost=result.transaction_cost,
                slippage_cost=result.slippage_cost,
                action=target_position,
                position_before=result.position_before,
                position_after=result.position_after,
                metadata=result.metadata,
            )
        )

        observation = np.asarray(self.observations[self._idx].observation_vector, dtype=np.float32)
        info = {"episode_id": self._episode_id, "step_result": result.model_dump(), "offline_only": True}
        if gym is not None:
            return observation, reward, done, False, info
        return observation, reward, done, info

    def _normalize_action(self, action: float | int | np.ndarray) -> float:
        if self.action_space_kind == "discrete":
            if isinstance(action, np.ndarray):
                action = int(action.item())
            return {-1: -1.0, 0: 0.0, 1: 1.0, 2: 1.0}.get(int(action), 0.0)
        if isinstance(action, np.ndarray):
            action = float(np.asarray(action).reshape(-1)[0])
        return float(np.clip(float(action), -1.0, 1.0))
