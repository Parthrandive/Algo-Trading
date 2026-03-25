from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from typing import Iterable

from src.agents.strategic.config import DeliberationConfig
from src.agents.strategic.policy_manager import PolicySnapshotManager
from src.agents.strategic.schemas import (
    DeliberationOutcome,
    PolicySnapshot,
    PolicySnapshotSource,
    QualityStatus,
    RiskMode,
    StrategicObservation,
    StudentPolicyArtifact,
)


class DeliberationEngine:
    """
    Slow Loop controller for snapshot refresh and rejection/bypass decisions.
    """

    def __init__(
        self,
        snapshot_manager: PolicySnapshotManager,
        *,
        config: DeliberationConfig | None = None,
    ) -> None:
        self.snapshot_manager = snapshot_manager
        self.config = config or DeliberationConfig()

    def run(
        self,
        observations: Iterable[StrategicObservation],
        *,
        candidate_artifact: StudentPolicyArtifact | None = None,
        stale_data_breach: bool = False,
        latency_breach: bool = False,
    ) -> DeliberationOutcome:
        observation_list = tuple(observations)
        now = datetime.now(UTC)
        if not observation_list:
            return DeliberationOutcome(
                timestamp=now,
                bypass_active=True,
                bypass_reason="no observations available",
                trade_rejections=("missing_observations",),
            )

        volatility_score = mean(max(abs(obs.var_95), abs(obs.es_95)) for obs in observation_list)
        bypass_active = volatility_score >= self.config.bypass_volatility_threshold
        bypass_reason = "high_volatility_bypass" if bypass_active else None
        trade_rejections = list(self._collect_trade_rejections(observation_list, stale_data_breach, latency_breach))

        selected_snapshot: PolicySnapshot | None = None
        if candidate_artifact is not None and not bypass_active:
            selected_snapshot = self.snapshot_manager.build_snapshot(
                artifact=candidate_artifact,
                source_type=PolicySnapshotSource.ASYNC_COMPLETION,
                generated_at=now,
                risk_mode=RiskMode.NORMAL if not trade_rejections else RiskMode.REDUCE_ONLY,
                quality_status=QualityStatus.PASS if not trade_rejections else QualityStatus.WARN,
                metadata={
                    "observation_count": len(observation_list),
                    "mean_volatility_score": volatility_score,
                },
            )

        return DeliberationOutcome(
            timestamp=now,
            symbol=observation_list[0].symbol,
            bypass_active=bypass_active,
            bypass_reason=bypass_reason,
            trade_rejections=tuple(trade_rejections),
            selected_snapshot=selected_snapshot,
            metadata={
                "mean_volatility_score": volatility_score,
                "slow_loop_target_latency_ms": self.config.target_latency_ms,
                "candidate_snapshot_ready": selected_snapshot is not None,
            },
        )

    def publish_if_allowed(self, outcome: DeliberationOutcome) -> PolicySnapshot | None:
        if outcome.bypass_active or outcome.selected_snapshot is None:
            return None
        self.snapshot_manager.publish(outcome.selected_snapshot)
        return outcome.selected_snapshot

    def _collect_trade_rejections(
        self,
        observations: tuple[StrategicObservation, ...],
        stale_data_breach: bool,
        latency_breach: bool,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if stale_data_breach:
            reasons.append("stale_data_ttl_breach")
        if latency_breach:
            reasons.append("latency_cap_breach")
        if any(obs.quality_status == "fail" for obs in observations):
            reasons.append("observation_quality_fail")
        if any(obs.crisis_mode for obs in observations):
            reasons.append("risk_control_breach")
        return tuple(reasons)
