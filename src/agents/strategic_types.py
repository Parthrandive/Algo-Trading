from __future__ import annotations

from enum import Enum


class ActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"


class RiskMode(str, Enum):
    NORMAL = "normal"
    REDUCE_ONLY = "reduce_only"
    CLOSE_ONLY = "close_only"
    KILL_SWITCH = "kill_switch"
