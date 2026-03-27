from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Mapping

from src.agents.strategic.config import ImpactMonitorConfig


class InstrumentBucket(str, Enum):
    LIQUID_LARGE_CAP = "liquid_large_cap"
    MID_LIQUIDITY = "mid_liquidity"
    USDINR = "usdinr"
    MCX_GOLD = "mcx_gold"


@dataclass(frozen=True)
class ImpactThreshold:
    participation_limit: float
    slippage_alert_bps: float


@dataclass(frozen=True)
class FillEvent:
    symbol: str
    bucket: InstrumentBucket
    quantity: int
    adv: float
    model_slippage_bps: float
    realized_slippage_bps: float
    timestamp: datetime


@dataclass(frozen=True)
class ImpactDecision:
    symbol: str
    bucket: InstrumentBucket
    breach: bool
    reasons: tuple[str, ...]
    size_multiplier: float
    participation_rate: float
    slippage_delta_bps: float
    impact_score: float
    cooldown_until: datetime | None
    risk_override: str | None
    event_type: str


@dataclass
class _SymbolState:
    size_multiplier: float = 1.0
    cooldown_until: datetime | None = None
    clean_fill_streak: int = 0


@dataclass
class _BucketMetrics:
    total_fills: int = 0
    breaches: int = 0
    auto_reduction_events: int = 0
    last_breach_at: datetime | None = None


def default_thresholds() -> dict[InstrumentBucket, ImpactThreshold]:
    return {
        InstrumentBucket.LIQUID_LARGE_CAP: ImpactThreshold(participation_limit=0.05, slippage_alert_bps=20.0),
        InstrumentBucket.MID_LIQUIDITY: ImpactThreshold(participation_limit=0.03, slippage_alert_bps=25.0),
        InstrumentBucket.USDINR: ImpactThreshold(participation_limit=0.08, slippage_alert_bps=20.0),
        InstrumentBucket.MCX_GOLD: ImpactThreshold(participation_limit=0.04, slippage_alert_bps=22.0),
    }


class ImpactSlippageMonitor:
    """
    Tier 1-A monitor for real-time impact/slippage with automatic sizing reductions.
    """

    def __init__(
        self,
        thresholds: Mapping[InstrumentBucket, ImpactThreshold] | None = None,
        *,
        config: ImpactMonitorConfig | None = None,
    ) -> None:
        self.thresholds = dict(thresholds or default_thresholds())
        self.config = config or ImpactMonitorConfig()
        self._symbol_state: dict[str, _SymbolState] = {}
        self._metrics: dict[InstrumentBucket, _BucketMetrics] = {
            bucket: _BucketMetrics() for bucket in self.thresholds
        }
        self._events: list[dict[str, object]] = []

    def evaluate_fill(self, fill: FillEvent) -> ImpactDecision:
        now = fill.timestamp.astimezone(UTC)
        threshold = self.thresholds[fill.bucket]
        state = self._symbol_state.setdefault(fill.symbol, _SymbolState())
        metrics = self._metrics[fill.bucket]

        participation_rate = abs(fill.quantity) / max(float(fill.adv), 1.0)
        slippage_delta = float(fill.realized_slippage_bps - fill.model_slippage_bps)
        impact_score = participation_rate * max(fill.realized_slippage_bps, 0.0) * 100.0

        reasons: list[str] = []
        if participation_rate > threshold.participation_limit:
            reasons.append("participation_limit_breach")
        if slippage_delta > threshold.slippage_alert_bps:
            reasons.append("slippage_delta_breach")

        breach = len(reasons) > 0
        metrics.total_fills += 1

        if breach:
            metrics.breaches += 1
            metrics.auto_reduction_events += 1
            metrics.last_breach_at = now
            state.clean_fill_streak = 0
            state.cooldown_until = now + timedelta(seconds=int(self.config.cooldown_seconds))
            state.size_multiplier = max(
                float(self.config.min_size_multiplier),
                state.size_multiplier * (1.0 - float(self.config.reduction_step_fraction)),
            )
            event_type = "IMPACT_BREACH"
            risk_override = "reduce_only"
        else:
            event_type = "IMPACT_OK"
            risk_override = "reduce_only" if state.size_multiplier < 1.0 else None
            if state.cooldown_until is not None and now < state.cooldown_until:
                state.clean_fill_streak = 0
            else:
                state.clean_fill_streak += 1
                if (
                    state.size_multiplier < 1.0
                    and state.clean_fill_streak >= int(self.config.hysteresis_clean_fills)
                ):
                    state.size_multiplier = min(
                        1.0,
                        state.size_multiplier + float(self.config.reduction_step_fraction),
                    )
                    state.clean_fill_streak = 0
                    event_type = "IMPACT_RESTORE_STEP"
                    risk_override = "reduce_only" if state.size_multiplier < 1.0 else None

        self._events.append(
            {
                "timestamp": now.isoformat(),
                "symbol": fill.symbol,
                "bucket": fill.bucket.value,
                "event_type": event_type,
                "size_multiplier": state.size_multiplier,
                "participation_rate": participation_rate,
                "slippage_delta_bps": slippage_delta,
                "impact_score": impact_score,
                "reasons": tuple(reasons),
            }
        )

        return ImpactDecision(
            symbol=fill.symbol,
            bucket=fill.bucket,
            breach=breach,
            reasons=tuple(reasons),
            size_multiplier=state.size_multiplier,
            participation_rate=participation_rate,
            slippage_delta_bps=slippage_delta,
            impact_score=impact_score,
            cooldown_until=state.cooldown_until,
            risk_override=risk_override,
            event_type=event_type,
        )

    def size_multiplier(self, symbol: str) -> float:
        return self._symbol_state.get(symbol, _SymbolState()).size_multiplier

    def recent_events(self, limit: int = 50) -> tuple[dict[str, object], ...]:
        return tuple(self._events[-max(limit, 0) :])

    def dashboard_snapshot(self) -> dict[str, object]:
        bucket_rows: list[dict[str, object]] = []
        for bucket, stats in self._metrics.items():
            breach_rate = (stats.breaches / stats.total_fills) if stats.total_fills else 0.0
            bucket_rows.append(
                {
                    "bucket": bucket.value,
                    "fills": stats.total_fills,
                    "breaches": stats.breaches,
                    "breach_rate": breach_rate,
                    "auto_reduction_events": stats.auto_reduction_events,
                    "last_breach_at": stats.last_breach_at.isoformat() if stats.last_breach_at else None,
                }
            )

        return {
            "updated_at": datetime.now(UTC).isoformat(),
            "bucket_metrics": bucket_rows,
            "active_reductions": {
                symbol: state.size_multiplier
                for symbol, state in self._symbol_state.items()
                if state.size_multiplier < 1.0
            },
        }
