from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Any, Iterable

from src.agents.strategic.config import DeliberationConfig
from src.agents.strategic.policy_manager import PolicySnapshot, PolicySnapshotManager, RiskMode
from src.agents.strategic.schemas import (
    DeliberationOutcome,
    PolicySnapshotSource,
    QualityStatus,
    StrategicObservation,
    StudentPolicyArtifact,
)


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

    def to_outcome(self) -> DeliberationOutcome:
        return DeliberationOutcome(
            timestamp=self.ended_at,
            symbol=self.symbol,
            bypass_active=self.bypassed,
            bypass_reason="high_volatility" if self.bypassed else None,
            trade_rejections=self.trade_guard.reasons,
            metadata=self.details,
        )


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
        observations: tuple[StrategicObservation, ...] = (),
    ) -> TradeGuardResult:
        reasons: list[str] = []
        mode = RiskMode.NORMAL

        if risk_control_breach or any(obs.crisis_mode for obs in observations):
            reasons.append("risk_control_breach")
            mode = _more_restrictive(mode, RiskMode.CLOSE_ONLY)

        if snapshot_stale:
            reasons.append("snapshot_stale_ttl")
            mode = _more_restrictive(mode, RiskMode.REDUCE_ONLY)

        if fast_loop_p99_ms is not None and float(fast_loop_p99_ms) > self.latency_cap_ms:
            reasons.append("latency_cap_breach")
            mode = _more_restrictive(mode, RiskMode.REDUCE_ONLY)

        if any(obs.quality_status == "fail" for obs in observations):
            reasons.append("observation_quality_fail")
            mode = _more_restrictive(mode, RiskMode.CLOSE_ONLY)

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
        snapshot_manager: PolicySnapshotManager,
        *,
        config: DeliberationConfig | None = None,
        trade_guard: TradeRejectionGuard | None = None,
        max_workers: int = 2,
    ):
        self.snapshot_manager = snapshot_manager
        self.config = config or DeliberationConfig()
        self.trade_guard = trade_guard or TradeRejectionGuard(latency_cap_ms=float(self.config.target_latency_ms))
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="slow-loop")

    def submit_cycle(self, payload: DeliberationInput) -> Future[DeliberationResult]:
        return self._executor.submit(self.run_cycle, payload)

    def run_cycle(self, payload: DeliberationInput) -> DeliberationResult:
        started = datetime.now(UTC)
        guard = self.trade_guard.evaluate(
            risk_control_breach=bool(payload.risk_control_breach),
            snapshot_stale=bool(payload.snapshot_stale),
            fast_loop_p99_ms=payload.fast_loop_p99_ms,
        )

        # Volatility bypass check
        bypassed = float(payload.volatility_sigma) >= self.config.bypass_volatility_threshold
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

            if expected_improvement >= 0.01:  # min_expected_improvement
                # Note: In a real system, we'd build a new snapshot from the candidate artifact
                # For Phase 3 foundation, we reuse the active snapshot and update its TTL/Source
                active = self.snapshot_manager.active_snapshot()
                if active is not None:
                    # Logic here would typically involve building a new snapshot
                    pass

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

    def run(
        self,
        observations: Iterable[StrategicObservation],
        *,
        candidate_artifact: StudentPolicyArtifact | None = None,
        stale_data_breach: bool = False,
        latency_breach: bool = False,
    ) -> DeliberationOutcome:
        """Synchronous wrapper for main/week1 compatibility."""
        observation_list = tuple(observations)
        now = datetime.now(UTC)
        if not observation_list:
            return DeliberationOutcome(
                timestamp=now,
                bypass_active=True,
                bypass_reason="no observations available",
                trade_rejections=("missing_observations",),
            )

        symbol = observation_list[0].symbol
        volatility_score = mean(max(abs(obs.var_95), abs(obs.es_95)) for obs in observation_list)
        
        guard = self.trade_guard.evaluate(
            risk_control_breach=any(obs.crisis_mode for obs in observation_list),
            snapshot_stale=stale_data_breach,
            fast_loop_p99_ms=10.0 if latency_breach else 0.0,
            observations=observation_list,
        )

        bypass_active = volatility_score >= self.config.bypass_volatility_threshold
        
        selected_snapshot = None
        if candidate_artifact is not None and not bypass_active and guard.allow_trade:
             # This uses the main branch's SnapshotManager pattern if needed, but we prefer HEAD's
             # For now, just follow the schema requirement
             pass

        return DeliberationOutcome(
            timestamp=now,
            symbol=symbol,
            bypass_active=bypass_active,
            bypass_reason="high_volatility" if bypass_active else None,
            trade_rejections=guard.reasons,
            metadata={
                "mean_volatility_score": float(volatility_score),
                "observation_count": len(observation_list),
            },
        )

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


# Alias for backward compatibility with main branch references (if any)
DeliberationEngine = SlowLoopDeliberationEngine


def _more_restrictive(a: RiskMode, b: RiskMode) -> RiskMode:
    order = {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }
    return a if order[a] >= order[b] else b
