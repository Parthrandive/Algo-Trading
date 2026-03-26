from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.agents.strategic.policy_manager import PolicySnapshot, PolicySnapshotManager, RiskMode


@dataclass(frozen=True)
class DeliberationConfig:
    volatility_bypass_sigma: float = 1.5
    latency_cap_ms: float = 10.0
    min_expected_improvement: float = 0.01
    snapshot_ttl: timedelta = timedelta(minutes=15)
    max_workers: int = 1

    def __post_init__(self) -> None:
        if float(self.volatility_bypass_sigma) <= 0.0:
            raise ValueError("volatility_bypass_sigma must be > 0")
        if float(self.latency_cap_ms) <= 0.0:
            raise ValueError("latency_cap_ms must be > 0")
        if self.snapshot_ttl.total_seconds() <= 0:
            raise ValueError("snapshot_ttl must be > 0")
        if int(self.max_workers) < 1:
            raise ValueError("max_workers must be >= 1")


@dataclass(frozen=True)
class DeliberationInput:
    symbol: str
    trigger: str
    input_snapshot_id: str | None
    volatility_sigma: float
    current_policy_score: float
    candidate_policy_score: float
    risk_control_breach: bool = False
    snapshot_stale: bool = False
    fast_loop_p99_ms: float | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeGuardResult:
    allow_trade: bool
    reasons: tuple[str, ...]
    mode: RiskMode


@dataclass(frozen=True)
class DeliberationResult:
    symbol: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    bypassed: bool
    triggered_refresh: bool
    input_snapshot_id: str | None
    output_snapshot_id: str | None
    trade_guard: TradeGuardResult
    details: dict[str, Any]


class TradeRejectionGuard:
    """Rejects trades on risk breach, stale snapshot, or latency-cap breach."""

    def __init__(self, *, latency_cap_ms: float = 10.0):
        if latency_cap_ms <= 0.0:
            raise ValueError("latency_cap_ms must be > 0")
        self.latency_cap_ms = float(latency_cap_ms)

    def evaluate(
        self,
        *,
        risk_control_breach: bool,
        snapshot_stale: bool,
        fast_loop_p99_ms: float | None,
    ) -> TradeGuardResult:
        reasons: list[str] = []
        mode = RiskMode.NORMAL

        if risk_control_breach:
            reasons.append("risk_control_breach")
            mode = _more_restrictive(mode, RiskMode.CLOSE_ONLY)

        if snapshot_stale:
            reasons.append("snapshot_stale_ttl")
            mode = _more_restrictive(mode, RiskMode.REDUCE_ONLY)

        if fast_loop_p99_ms is not None and float(fast_loop_p99_ms) > self.latency_cap_ms:
            reasons.append("latency_cap_breach")
            mode = _more_restrictive(mode, RiskMode.REDUCE_ONLY)

        # Pending simulation is intentionally not a rejection reason.
        return TradeGuardResult(
            allow_trade=len(reasons) == 0,
            reasons=tuple(reasons),
            mode=mode,
        )


class SlowLoopDeliberationEngine:
    """
    Week 2 Day 4 asynchronous deliberation engine.

    Slow Loop runs off the critical path and never blocks Fast Loop order flow.
    """

    def __init__(
        self,
        *,
        snapshot_manager: PolicySnapshotManager,
        config: DeliberationConfig | None = None,
        trade_guard: TradeRejectionGuard | None = None,
    ):
        self.snapshot_manager = snapshot_manager
        self.config = config or DeliberationConfig()
        self.trade_guard = trade_guard or TradeRejectionGuard(latency_cap_ms=self.config.latency_cap_ms)
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="slow-loop")

    def submit_cycle(self, payload: DeliberationInput) -> Future[DeliberationResult]:
        return self._executor.submit(self.run_cycle, payload)

    def run_cycle(self, payload: DeliberationInput) -> DeliberationResult:
        started = datetime.now(UTC)
        guard = self.trade_guard.evaluate(
            risk_control_breach=bool(payload.risk_control_breach),
            snapshot_stale=bool(payload.snapshot_stale),
            fast_loop_p99_ms=payload.fast_loop_p99_ms,
        )

        bypassed = float(payload.volatility_sigma) >= self.config.volatility_bypass_sigma
        details: dict[str, Any] = {
            "trigger": payload.trigger,
            "volatility_sigma": float(payload.volatility_sigma),
            "current_policy_score": float(payload.current_policy_score),
            "candidate_policy_score": float(payload.candidate_policy_score),
            "risk_control_breach": bool(payload.risk_control_breach),
            "snapshot_stale": bool(payload.snapshot_stale),
            "fast_loop_p99_ms": payload.fast_loop_p99_ms,
        }
        details.update(payload.context)

        triggered_refresh = False
        output_snapshot_id: str | None = None

        if not guard.allow_trade:
            self.snapshot_manager.force_mode(guard.mode, reason=",".join(guard.reasons))
        elif bypassed:
            details["bypass_reason"] = "high_volatility"
        else:
            expected_improvement = float(payload.candidate_policy_score - payload.current_policy_score)
            details["expected_improvement"] = expected_improvement

            if expected_improvement >= self.config.min_expected_improvement:
                active = self.snapshot_manager.active_snapshot()
                if active is not None:
                    published_at = datetime.now(UTC)
                    refreshed = PolicySnapshot(
                        snapshot_id=f"{payload.symbol}:{published_at.strftime('%Y%m%dT%H%M%S.%fZ')}",
                        policy_id=active.policy_id,
                        policy_type=active.policy_type,
                        generated_at=published_at,
                        expires_at=published_at + self.config.snapshot_ttl,
                        schema_version=active.schema_version,
                        quality_status="pass",
                        source_type=f"slow_loop:{payload.trigger}",
                        metadata={
                            "input_snapshot_id": payload.input_snapshot_id,
                            "expected_improvement": expected_improvement,
                            "deliberation_type": "policy_refresh",
                        },
                    )
                    self.snapshot_manager.publish_snapshot(
                        refreshed,
                        trigger=payload.trigger,
                        emergency=False,
                    )
                    triggered_refresh = True
                    output_snapshot_id = refreshed.snapshot_id
                else:
                    details["refresh_skipped"] = "no_active_snapshot"
            else:
                details["refresh_skipped"] = "insufficient_improvement"

        ended = datetime.now(UTC)
        duration_ms = (ended - started).total_seconds() * 1_000.0
        return DeliberationResult(
            symbol=payload.symbol,
            started_at=started,
            ended_at=ended,
            duration_ms=float(duration_ms),
            bypassed=bypassed,
            triggered_refresh=triggered_refresh,
            input_snapshot_id=payload.input_snapshot_id,
            output_snapshot_id=output_snapshot_id,
            trade_guard=guard,
            details=details,
        )

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


def _more_restrictive(a: RiskMode, b: RiskMode) -> RiskMode:
    order = {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }
    return a if order[a] >= order[b] else b
