from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class CompositeRewardConfig:
    volatility_penalty: float = 0.25
    drawdown_penalty: float = 0.35
    turnover_penalty: float = 0.02
    crisis_penalty: float = 0.10
    divergence_penalty: float = 0.05
    max_abs_reward: float = 1.0
    annualization_factor: int = 252


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    net_return: float
    volatility_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    turnover_penalty: float = 0.0
    crisis_penalty: float = 0.0
    divergence_penalty: float = 0.0

    def components(self) -> dict[str, float]:
        return {
            "net_return": float(self.net_return),
            "volatility_penalty": float(self.volatility_penalty),
            "drawdown_penalty": float(self.drawdown_penalty),
            "turnover_penalty": float(self.turnover_penalty),
            "crisis_penalty": float(self.crisis_penalty),
            "divergence_penalty": float(self.divergence_penalty),
            "reward": float(self.reward),
        }


def step_return_reward(gross_return: float, transaction_cost: float = 0.0, slippage_cost: float = 0.0) -> float:
    return float(gross_return - transaction_cost - slippage_cost)


def ra_drl_step_reward(
    *,
    gross_return: float,
    transaction_cost: float,
    slippage_cost: float,
    turnover: float,
    running_returns: Sequence[float],
    current_drawdown: float,
    crisis_mode: bool = False,
    agent_divergence: bool = False,
    config: CompositeRewardConfig | None = None,
) -> RewardBreakdown:
    cfg = config or CompositeRewardConfig()
    net_return = step_return_reward(gross_return, transaction_cost, slippage_cost)
    trailing = _as_array(running_returns)
    realized_vol = float(np.std(trailing, ddof=0)) if trailing.size else 0.0
    volatility_penalty = cfg.volatility_penalty * realized_vol
    drawdown_penalty = cfg.drawdown_penalty * max(0.0, float(current_drawdown))
    turnover_penalty = cfg.turnover_penalty * max(0.0, float(turnover))
    crisis_penalty = cfg.crisis_penalty if crisis_mode else 0.0
    divergence_penalty = cfg.divergence_penalty if agent_divergence else 0.0
    reward = net_return - (
        volatility_penalty
        + drawdown_penalty
        + turnover_penalty
        + crisis_penalty
        + divergence_penalty
    )
    clipped = float(np.clip(reward, -cfg.max_abs_reward, cfg.max_abs_reward))
    return RewardBreakdown(
        reward=clipped,
        net_return=net_return,
        volatility_penalty=volatility_penalty,
        drawdown_penalty=drawdown_penalty,
        turnover_penalty=turnover_penalty,
        crisis_penalty=crisis_penalty,
        divergence_penalty=divergence_penalty,
    )


def sharpe_ratio(returns: Sequence[float], annualization_factor: int = 252, risk_free_rate: float = 0.0) -> float:
    values = _as_array(returns)
    if values.size == 0:
        return 0.0
    excess = values - (risk_free_rate / max(annualization_factor, 1))
    std = float(np.std(excess, ddof=0))
    if std <= 0.0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(annualization_factor))


def sortino_ratio(returns: Sequence[float], annualization_factor: int = 252, target_return: float = 0.0) -> float:
    values = _as_array(returns)
    if values.size == 0:
        return 0.0
    downside = values[values < target_return]
    if downside.size == 0:
        return float(np.mean(values) * math.sqrt(annualization_factor))
    downside_deviation = float(np.sqrt(np.mean(np.square(downside - target_return))))
    if downside_deviation <= 0.0:
        return 0.0
    return float((np.mean(values) - target_return) / downside_deviation * math.sqrt(annualization_factor))


def calmar_ratio(returns: Sequence[float], annualization_factor: int = 252) -> float:
    values = _as_array(returns)
    if values.size == 0:
        return 0.0
    drawdown = max_drawdown(values)
    if drawdown <= 0.0:
        return 0.0
    annualized_return = float(np.mean(values) * annualization_factor)
    return annualized_return / drawdown


def ra_drl_reward(returns: Sequence[float], volatility_penalty: float = 0.5, drawdown_penalty: float = 0.25) -> float:
    values = _as_array(returns)
    if values.size == 0:
        return 0.0
    return float(
        np.mean(values)
        - (volatility_penalty * np.std(values, ddof=0))
        - (drawdown_penalty * max_drawdown(values))
    )


def kelly_reward(win_probability: float, win_loss_ratio: float, action_scale: float = 1.0) -> float:
    if win_loss_ratio <= 0.0:
        return 0.0
    fraction = win_probability - ((1.0 - win_probability) / win_loss_ratio)
    return float(max(-1.0, min(1.0, fraction * action_scale)))


def max_drawdown(returns: Sequence[float]) -> float:
    values = _as_array(returns)
    if values.size == 0:
        return 0.0
    equity_curve = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = 1.0 - (equity_curve / peaks)
    return float(np.max(drawdowns))


def win_rate(returns: Sequence[float]) -> float:
    values = _as_array(returns)
    if values.size == 0:
        return 0.0
    return float(np.mean(values > 0.0))


def trading_performance_summary(returns: Sequence[float], annualization_factor: int = 252) -> dict[str, float]:
    values = _as_array(returns)
    if values.size == 0:
        return {
            "cumulative_return": 0.0,
            "mean_step_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "ra_drl_objective": 0.0,
            "steps": 0.0,
        }
    cumulative = float(np.prod(1.0 + values) - 1.0)
    return {
        "cumulative_return": cumulative,
        "mean_step_return": float(np.mean(values)),
        "sharpe": sharpe_ratio(values, annualization_factor=annualization_factor),
        "sortino": sortino_ratio(values, annualization_factor=annualization_factor),
        "calmar": calmar_ratio(values, annualization_factor=annualization_factor),
        "max_drawdown": max_drawdown(values),
        "win_rate": win_rate(values),
        "ra_drl_objective": ra_drl_reward(values),
        "steps": float(values.size),
    }


def _as_array(values: Sequence[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float)
    return np.asarray(list(values), dtype=float)
