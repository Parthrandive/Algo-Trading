from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from src.agents.strategic.config import EnvironmentCostConfig
from src.agents.strategic.reward import CompositeRewardConfig, RewardBreakdown, ra_drl_step_reward, step_return_reward
from src.agents.strategic.schemas import RewardLog, StepResult, StrategicObservation

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None


class _FallbackEnv:
    """Small stand-in so foundation tests work without gymnasium installed."""


@dataclass(frozen=True)
class EnvironmentRiskConfig:
    normal_position_cap: float = 1.0
    crisis_position_cap: float = 0.30
    divergence_position_cap: float = 0.50
    warn_quality_position_cap: float = 0.70
    fail_quality_position_cap: float = 0.0


@dataclass
class StrategicTradingEnv(_FallbackEnv if gym is None else gym.Env):
    observations: list[StrategicObservation]
    prices: list[float]
    reward_name: str = "step_return"
    cost_config: EnvironmentCostConfig = field(default_factory=EnvironmentCostConfig)
    reward_config: CompositeRewardConfig = field(default_factory=CompositeRewardConfig)
    risk_config: EnvironmentRiskConfig = field(default_factory=EnvironmentRiskConfig)
    action_space_kind: str = "continuous"

    def __post_init__(self) -> None:
        if len(self.observations) < 2:
            raise ValueError("StrategicTradingEnv requires at least two observations")
        if len(self.prices) != len(self.observations):
            raise ValueError("prices and observations must have the same length")
        if self.reward_name not in {"step_return", "ra_drl_composite"}:
            raise ValueError("reward_name must be one of {'step_return', 'ra_drl_composite'}")
        self._episode_id = ""
        self._idx = 0
        self._position = 0.0
        self._portfolio_value = float(self.cost_config.initial_cash)
        self._net_returns: list[float] = []
        self._equity_curve: list[float] = [self._portfolio_value]
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
        self._net_returns = []
        self._equity_curve = [self._portfolio_value]
        self._episode_id = f"strategic-{uuid4().hex[:12]}"
        self.reward_logs = []
        obs = np.asarray(self.observations[self._idx].observation_vector, dtype=np.float32)
        return obs, {"episode_id": self._episode_id, "offline_only": True}

    def step(self, action: float | int | np.ndarray):
        if self._idx >= len(self.observations) - 1:
            raise RuntimeError("Environment already completed. Call reset() before stepping again.")

        observation = self.observations[self._idx]
        target_position = self._normalize_action(action)
        target_position, position_cap = self._apply_risk_caps(target_position, observation)
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
        net_return = step_return_reward(gross_return, transaction_cost, slippage_cost)
        next_portfolio_value = self._portfolio_value * (1.0 + net_return)
        next_drawdown = self._drawdown_from_equity(self._equity_curve + [next_portfolio_value])

        breakdown = self._compute_reward_breakdown(
            gross_return=gross_return,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            turnover=turnover,
            net_return=net_return,
            current_drawdown=next_drawdown,
            observation=observation,
        )
        reward = breakdown.reward

        self._portfolio_value = next_portfolio_value
        self._position = target_position
        self._net_returns.append(net_return)
        self._equity_curve.append(self._portfolio_value)
        self._idx += 1
        done = self._idx >= len(self.observations) - 1

        result = StepResult(
            reward=reward,
            gross_return=gross_return,
            net_return=net_return,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            position_before=position_before,
            position_after=self._position,
            portfolio_value=self._portfolio_value,
            done=done,
            metadata={
                "price_return": price_return,
                "reward_components": breakdown.components(),
                "position_cap": position_cap,
                "reward_name": self.reward_name,
            },
        )
        self.reward_logs.append(
            RewardLog(
                symbol=observation.symbol,
                timestamp=observation.timestamp,
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

        next_observation = np.asarray(self.observations[self._idx].observation_vector, dtype=np.float32)
        info = {"episode_id": self._episode_id, "step_result": result.model_dump(), "offline_only": True}
        if gym is not None:
            return next_observation, reward, done, False, info
        return next_observation, reward, done, info

    def _normalize_action(self, action: float | int | np.ndarray) -> float:
        if self.action_space_kind == "discrete":
            if isinstance(action, np.ndarray):
                action = int(action.item())
            mapping = {
                -1: -1.0,  # legacy sell
                0: -1.0,  # sell
                1: 0.0,  # hold
                2: 1.0,  # buy
            }
            return mapping.get(int(action), 0.0)
        if isinstance(action, np.ndarray):
            action = float(np.asarray(action).reshape(-1)[0])
        return float(np.clip(float(action), -1.0, 1.0))

    def _compute_reward_breakdown(
        self,
        *,
        gross_return: float,
        transaction_cost: float,
        slippage_cost: float,
        turnover: float,
        net_return: float,
        current_drawdown: float,
        observation: StrategicObservation,
    ) -> RewardBreakdown:
        if self.reward_name == "step_return":
            return RewardBreakdown(reward=net_return, net_return=net_return)
        return ra_drl_step_reward(
            gross_return=gross_return,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            turnover=turnover,
            running_returns=[*self._net_returns, net_return],
            current_drawdown=current_drawdown,
            crisis_mode=bool(observation.crisis_mode),
            agent_divergence=bool(observation.agent_divergence),
            config=self.reward_config,
        )

    def _apply_risk_caps(self, position: float, observation: StrategicObservation) -> tuple[float, float]:
        cap = float(self.risk_config.normal_position_cap)
        if observation.crisis_mode:
            cap = min(cap, float(self.risk_config.crisis_position_cap))
        if observation.agent_divergence:
            cap = min(cap, float(self.risk_config.divergence_position_cap))
        quality = observation.quality_status.strip().lower()
        if quality == "warn":
            cap = min(cap, float(self.risk_config.warn_quality_position_cap))
        elif quality == "fail":
            cap = min(cap, float(self.risk_config.fail_quality_position_cap))
        cap = float(np.clip(cap, 0.0, 1.0))
        return float(np.clip(position, -cap, cap)), cap

    @staticmethod
    def _drawdown_from_equity(equity_curve: list[float]) -> float:
        if not equity_curve:
            return 0.0
        values = np.asarray(equity_curve, dtype=float)
        peaks = np.maximum.accumulate(values)
        drawdowns = 1.0 - (values / np.maximum(peaks, 1e-12))
        return float(np.max(drawdowns))
