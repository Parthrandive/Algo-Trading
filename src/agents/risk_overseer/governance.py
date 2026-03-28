from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from src.agents.risk_overseer.config import RiskOverseerConfig
from src.agents.strategic.risk_budgets import VolatilityScaledRiskBudgetEngine
from src.agents.strategic.schemas import RiskMode
from src.agents.strategic.xai_attribution import PnLAttributionEngine, XAILogger


@dataclass(frozen=True)
class DriftObservation:
    timestamp: datetime
    policy_id: str
    input_drift_score: float
    policy_drift_score: float
    provenance_reliability: float


@dataclass(frozen=True)
class DriftAlert:
    timestamp: datetime
    policy_id: str
    drift_score: float
    drift_threshold: float
    consecutive_breaches: int
    sustained_drift: bool
    demotion_triggered: bool
    recommended_risk_mode: RiskMode
    exposure_cap_multiplier: float
    provenance_reliability: float


@dataclass(frozen=True)
class FalseTriggerReview:
    timestamp: datetime
    false_trigger_rate: float
    acceptance_limit: float
    auto_adjustment_paused: bool
    escalate_to_manual_review: bool


@dataclass(frozen=True)
class Phase4RiskGateAudit:
    timestamp: datetime
    passed: bool
    checks: dict[str, bool]
    blocker_reasons: tuple[str, ...]
    xai_coverage: float
    drift_monitoring_active: bool
    false_trigger_rate: float
    false_trigger_manual_review_required: bool
    pnl_dashboard_live: bool
    surprise_kill_switch_passed: bool


class DriftSurveillanceMonitor:
    """
    Week 4 ADDM-like drift surveillance.

    The monitor keeps continuous drift state and escalates sustained breaches to
    automatic demotion guidance while scaling exposure dynamically with
    provenance reliability instead of fixed haircuts.
    """

    def __init__(self, config: RiskOverseerConfig | None = None) -> None:
        self.config = config or RiskOverseerConfig()
        self._consecutive_breaches: dict[str, int] = {}
        self._alerts: list[DriftAlert] = []

    def evaluate(self, observation: DriftObservation) -> DriftAlert:
        timestamp = _to_utc(observation.timestamp)
        drift_score = max(0.0, float(observation.input_drift_score), float(observation.policy_drift_score))
        threshold = float(self.config.drift_alert_threshold)

        breaches = self._consecutive_breaches.get(observation.policy_id, 0)
        if drift_score >= threshold:
            breaches += 1
        else:
            breaches = 0
        self._consecutive_breaches[observation.policy_id] = breaches

        sustained = breaches >= int(self.config.drift_sustained_breach_count)
        demotion_triggered = sustained

        if drift_score >= float(self.config.drift_close_only_threshold):
            mode = RiskMode.CLOSE_ONLY
        elif drift_score >= threshold:
            mode = RiskMode.REDUCE_ONLY
        else:
            mode = RiskMode.NORMAL

        exposure_cap_multiplier = self._dynamic_exposure_multiplier(
            drift_score=drift_score,
            provenance_reliability=float(observation.provenance_reliability),
        )
        if demotion_triggered:
            exposure_cap_multiplier = min(exposure_cap_multiplier, 0.25)

        alert = DriftAlert(
            timestamp=timestamp,
            policy_id=observation.policy_id,
            drift_score=drift_score,
            drift_threshold=threshold,
            consecutive_breaches=breaches,
            sustained_drift=sustained,
            demotion_triggered=demotion_triggered,
            recommended_risk_mode=mode,
            exposure_cap_multiplier=exposure_cap_multiplier,
            provenance_reliability=max(0.0, min(1.0, float(observation.provenance_reliability))),
        )
        self._alerts.append(alert)
        return alert

    def recent_alerts(self, limit: int = 100) -> tuple[DriftAlert, ...]:
        return tuple(self._alerts[-max(0, limit) :])

    def monitoring_active(self) -> bool:
        return len(self._alerts) > 0

    def _dynamic_exposure_multiplier(self, *, drift_score: float, provenance_reliability: float) -> float:
        reliability = max(0.0, min(1.0, provenance_reliability))
        provenance_component = reliability ** float(self.config.provenance_exposure_power)
        drift_penalty = max(0.0, 1.0 - min(1.0, drift_score))
        dynamic_multiplier = provenance_component * drift_penalty
        return max(float(self.config.provenance_exposure_floor), min(1.0, dynamic_multiplier))


class RiskFalseTriggerTracker:
    """Week 4 review wrapper around Tier 1-B false-trigger telemetry."""

    def __init__(
        self,
        risk_budget_engine: VolatilityScaledRiskBudgetEngine,
        config: RiskOverseerConfig | None = None,
    ) -> None:
        self.risk_budget_engine = risk_budget_engine
        self.config = config or RiskOverseerConfig()

    def review(self, *, timestamp: datetime | None = None) -> FalseTriggerReview:
        now = _to_utc(timestamp or datetime.now(UTC))
        rate = float(self.risk_budget_engine.false_trigger_rate(now=now))
        paused = bool(self.risk_budget_engine.auto_adjustment_paused(now=now))
        return FalseTriggerReview(
            timestamp=now,
            false_trigger_rate=rate,
            acceptance_limit=float(self.config.false_trigger_acceptance_limit),
            auto_adjustment_paused=paused,
            escalate_to_manual_review=paused or rate > float(self.config.false_trigger_acceptance_limit),
        )


class Phase4RiskGateAuditor:
    """Final Risk Overseer gate review for Week 4."""

    def __init__(self, config: RiskOverseerConfig | None = None) -> None:
        self.config = config or RiskOverseerConfig()

    def audit(
        self,
        *,
        drift_monitor: DriftSurveillanceMonitor,
        false_trigger_review: FalseTriggerReview,
        xai_logger: XAILogger,
        pnl_attribution: PnLAttributionEngine,
        surprise_kill_switch_passed: bool,
        drift_demotion_tested: bool,
    ) -> Phase4RiskGateAudit:
        pnl_snapshot = pnl_attribution.dashboard_snapshot()
        xai_coverage = float(xai_logger.coverage())
        checks = {
            "drift_monitoring_active": drift_monitor.monitoring_active(),
            "drift_demotion_tested": bool(drift_demotion_tested),
            "xai_coverage": xai_coverage >= float(self.config.xai_min_coverage),
            "pnl_dashboard_live": int(pnl_snapshot.get("events_count", 0)) > 0,
            "false_trigger_escalation_resolved": not false_trigger_review.escalate_to_manual_review,
            "surprise_kill_switch": bool(surprise_kill_switch_passed),
        }
        blockers = tuple(key for key, passed in checks.items() if not passed)
        return Phase4RiskGateAudit(
            timestamp=datetime.now(UTC),
            passed=len(blockers) == 0,
            checks=checks,
            blocker_reasons=blockers,
            xai_coverage=xai_coverage,
            drift_monitoring_active=checks["drift_monitoring_active"],
            false_trigger_rate=float(false_trigger_review.false_trigger_rate),
            false_trigger_manual_review_required=bool(false_trigger_review.escalate_to_manual_review),
            pnl_dashboard_live=checks["pnl_dashboard_live"],
            surprise_kill_switch_passed=bool(surprise_kill_switch_passed),
        )


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


__all__ = [
    "DriftAlert",
    "DriftObservation",
    "DriftSurveillanceMonitor",
    "FalseTriggerReview",
    "Phase4RiskGateAudit",
    "Phase4RiskGateAuditor",
    "RiskFalseTriggerTracker",
]
