from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping

from src.agents.strategic.config import XAIConfig


@dataclass(frozen=True)
class TradeExplanation:
    trade_id: str
    symbol: str
    timestamp: datetime
    top_feature_contributions: tuple[tuple[str, float], ...]
    agent_contributions: tuple[tuple[str, float], ...]
    signal_family_contributions: tuple[tuple[str, float], ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class AttributionTotals:
    by_agent: dict[str, float]
    by_signal_family: dict[str, float]
    by_symbol: dict[str, float]
    by_sector: dict[str, float]
    total_pnl: float


class XAILogger:
    """
    Day 6 SHAP-like top-k logging for trade decisions.
    """

    def __init__(self, config: XAIConfig | None = None) -> None:
        self.config = config or XAIConfig()
        self._seen_trades: set[str] = set()
        self._explained_trades: set[str] = set()
        self._records: list[TradeExplanation] = []

    def mark_trade_seen(self, trade_id: str) -> None:
        self._seen_trades.add(trade_id)

    def log_trade(
        self,
        *,
        trade_id: str,
        symbol: str,
        feature_contributions: Mapping[str, float],
        agent_contributions: Mapping[str, float],
        signal_family_contributions: Mapping[str, float],
        timestamp: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TradeExplanation:
        self._seen_trades.add(trade_id)
        top_k = int(self.config.top_k_features)
        ranked = sorted(feature_contributions.items(), key=lambda item: abs(float(item[1])), reverse=True)
        top_features = tuple((name, float(value)) for name, value in ranked[:top_k])

        record = TradeExplanation(
            trade_id=trade_id,
            symbol=symbol,
            timestamp=(timestamp or datetime.now(UTC)).astimezone(UTC),
            top_feature_contributions=top_features,
            agent_contributions=tuple((k, float(v)) for k, v in agent_contributions.items()),
            signal_family_contributions=tuple((k, float(v)) for k, v in signal_family_contributions.items()),
            metadata=dict(metadata or {}),
        )
        self._records.append(record)
        self._explained_trades.add(trade_id)
        return record

    def coverage(self) -> float:
        if not self._seen_trades:
            return 0.0
        return len(self._explained_trades) / len(self._seen_trades)

    def coverage_ok(self) -> bool:
        return self.coverage() >= float(self.config.min_coverage)

    def recent_records(self, limit: int = 100) -> tuple[TradeExplanation, ...]:
        return tuple(self._records[-max(0, limit) :])


class PnLAttributionEngine:
    """
    Aggregates live PnL attribution across agents and signal families.
    """

    def __init__(self) -> None:
        self._by_agent: defaultdict[str, float] = defaultdict(float)
        self._by_signal_family: defaultdict[str, float] = defaultdict(float)
        self._by_symbol: defaultdict[str, float] = defaultdict(float)
        self._by_sector: defaultdict[str, float] = defaultdict(float)
        self._events: list[dict[str, object]] = []

    def add_event(
        self,
        *,
        trade_id: str,
        symbol: str,
        sector: str,
        agent: str,
        signal_family: str,
        realized_pnl: float,
        timestamp: datetime | None = None,
    ) -> None:
        pnl = float(realized_pnl)
        self._by_agent[agent] += pnl
        self._by_signal_family[signal_family] += pnl
        self._by_symbol[symbol] += pnl
        self._by_sector[sector] += pnl
        self._events.append(
            {
                "trade_id": trade_id,
                "timestamp": (timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
                "symbol": symbol,
                "sector": sector,
                "agent": agent,
                "signal_family": signal_family,
                "realized_pnl": pnl,
            }
        )

    def totals(self) -> AttributionTotals:
        total_pnl = sum(self._by_symbol.values())
        return AttributionTotals(
            by_agent=dict(self._by_agent),
            by_signal_family=dict(self._by_signal_family),
            by_symbol=dict(self._by_symbol),
            by_sector=dict(self._by_sector),
            total_pnl=total_pnl,
        )

    def dashboard_snapshot(self) -> dict[str, object]:
        totals = self.totals()
        return {
            "updated_at": datetime.now(UTC).isoformat(),
            "total_pnl": totals.total_pnl,
            "by_agent": totals.by_agent,
            "by_signal_family": totals.by_signal_family,
            "by_symbol": totals.by_symbol,
            "by_sector": totals.by_sector,
            "events_count": len(self._events),
        }


class OperationalMetricsBoard:
    """
    Day 6 operational dashboard metrics:
    staleness, lag, mode-switches, OOD and kill-switch quality, MTTR.
    """

    def __init__(self) -> None:
        self._decision_staleness_seconds: list[float] = []
        self._feature_lag_seconds: list[float] = []
        self._mode_switch_count = 0
        self._ood_trigger_count = 0
        self._kill_switch_false_positives = 0
        self._mttr_seconds: list[float] = []

    def add_decision_staleness(self, seconds: float) -> None:
        self._decision_staleness_seconds.append(max(0.0, float(seconds)))

    def add_feature_lag(self, seconds: float) -> None:
        self._feature_lag_seconds.append(max(0.0, float(seconds)))

    def increment_mode_switch(self) -> None:
        self._mode_switch_count += 1

    def increment_ood_trigger(self) -> None:
        self._ood_trigger_count += 1

    def increment_kill_switch_false_positive(self) -> None:
        self._kill_switch_false_positives += 1

    def record_mttr(self, seconds: float) -> None:
        self._mttr_seconds.append(max(0.0, float(seconds)))

    def snapshot(self) -> dict[str, object]:
        def _mean(values: list[float]) -> float:
            if not values:
                return 0.0
            return sum(values) / len(values)

        return {
            "updated_at": datetime.now(UTC).isoformat(),
            "decision_staleness_avg_s": _mean(self._decision_staleness_seconds),
            "feature_lag_avg_s": _mean(self._feature_lag_seconds),
            "mode_switch_frequency": self._mode_switch_count,
            "ood_trigger_rate": self._ood_trigger_count,
            "kill_switch_false_positives": self._kill_switch_false_positives,
            "mttr_avg_s": _mean(self._mttr_seconds),
        }
