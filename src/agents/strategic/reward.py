from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def step_return_reward(gross_return: float, transaction_cost: float = 0.0, slippage_cost: float = 0.0) -> float:
    return float(gross_return - transaction_cost - slippage_cost)


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


def ra_drl_reward(returns: Sequence[float], volatility_penalty: float = 0.5) -> float:
    values = _as_array(returns)
    if values.size == 0:
        return 0.0
    return float(np.mean(values) - volatility_penalty * np.std(values, ddof=0))


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


def _as_array(values: Sequence[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float)
    return np.asarray(list(values), dtype=float)
