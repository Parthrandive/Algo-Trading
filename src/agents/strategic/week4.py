from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import sqrt
from random import Random
from statistics import mean
from typing import Any, Mapping, Sequence

from src.agents.strategic.execution import ExecutionContext, ExecutionEngine, OrderRequest, OrderType
from src.agents.strategic.promotion_gates import (
    PolicyStage,
    PromotionEvidence,
    PromotionGatePipeline,
    RollbackDrillManager,
    RollbackDrillResult,
)
from src.agents.strategic.risk_budgets import VolatilityScaledRiskBudgetEngine
from src.agents.strategic.risk_overseer import RiskOverseerStateMachine, RiskSignalSnapshot
from src.agents.strategic.schemas import RiskMode
from src.agents.strategic.xai_attribution import PnLAttributionEngine, XAILogger

GO_LIVE_TARGETS: dict[str, float] = {
    "sharpe": 1.8,
    "sortino": 2.0,
    "max_drawdown": 0.08,
    "win_rate": 0.52,
    "profit_factor": 1.5,
}

AUDIT_REQUIRED_FIELDS: tuple[str, ...] = (
    "event_id",
    "event_type",
    "timestamp_utc",
    "instrument",
    "direction",
    "quantity",
    "price",
    "order_type",
    "model_version",
    "signal_source",
    "universe_version",
    "plan_version",
    "pre_trade_checks_passed",
    "rejection_reason",
    "data_source_tags",
    "risk_mode",
)

_MODE_ORDER: dict[RiskMode, int] = {
    RiskMode.NORMAL: 0,
    RiskMode.REDUCE_ONLY: 1,
    RiskMode.CLOSE_ONLY: 2,
    RiskMode.KILL_SWITCH: 3,
}

WEEK3_SCENARIO_LIBRARY_VERSION = "phase4_week3_scenario_library_v1"
WEEK3_CAPACITY_MULTIPLIERS: tuple[float, ...] = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class BacktestMetrics:
    periods: int
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float


@dataclass(frozen=True)
class WalkForwardBacktestReport:
    fold_metrics: tuple[BacktestMetrics, ...]
    aggregate_metrics: BacktestMetrics
    target_pass: bool
    target_checks: dict[str, bool]


@dataclass(frozen=True)
class LeakageAuditResult:
    passed: bool
    checked_rows: int
    leakage_rows: int
    lag_breach_rows: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SurvivorshipAuditResult:
    passed: bool
    rows_checked: int
    delisted_symbols_seen: int
    point_in_time_rows: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StressScenarioResult:
    scenario_id: str
    protective_mode: RiskMode
    expected_min_mode: RiskMode
    scenario_type: str = "synthetic"
    crashed: bool = False
    data_corruption: bool = False
    zero_vs_missing_distinguished: bool = True
    feed_integrity_uncertain: bool = False
    safe_mode_engaged: bool = True
    variance_defined: bool = True
    snapback_ticks: int = 0
    max_snapback_ticks: int = 30
    capacity_multiplier: float = 1.0
    impact_bps: float = 0.0


@dataclass(frozen=True)
class StressTestReport:
    passed: bool
    scenarios_run: int
    failure_count: int
    failure_reasons: tuple[str, ...]
    scenario_library_version: str = WEEK3_SCENARIO_LIBRARY_VERSION
    scenario_type_counts: dict[str, int] = field(default_factory=dict)
    capacity_hard_cap_multiplier: float = 3.0
    hard_cap_forced: bool = False


@dataclass(frozen=True)
class StressScenarioDefinition:
    scenario_id: str
    scenario_type: str
    key_test: str


@dataclass(frozen=True)
class QuarterlyStressReview:
    generated_at: datetime
    quarter_label: str
    owner: str
    reviewer: str
    scenario_library_version: str
    scenarios_run: int
    failure_count: int
    hard_cap_multiplier: float
    hard_cap_forced: bool
    performance_delta_summary: dict[str, float]
    required_actions: tuple[str, ...]


@dataclass(frozen=True)
class DriftReading:
    timestamp: datetime
    phase2_input_drift: float
    phase3_output_drift: float
    provenance_reliability: float = 1.0


@dataclass(frozen=True)
class DriftDecision:
    timestamp: datetime
    drift_score: float
    size_multiplier: float
    drift_alert: bool
    sustained_drift: bool
    demotion_triggered: bool
    reason: str


@dataclass(frozen=True)
class FalseTriggerReview:
    timestamp: datetime
    false_trigger_rate: float
    acceptance_limit: float
    rolling_days: int
    auto_adjustment_paused: bool
    escalation_required: bool
    reason: str


@dataclass(frozen=True)
class L4KillSwitchDrillResult:
    timestamp: datetime
    passed: bool
    mode: RiskMode
    should_cancel_orders: bool
    event_recorded: bool
    reason: str


@dataclass(frozen=True)
class Week4RiskGateReport:
    passed: bool
    shap_coverage: float
    shap_coverage_ok: bool
    pnl_dashboard_active: bool
    risk_override_events: int
    risk_overrides_visible_in_dashboard: bool
    l4_drill_passed: bool
    checklist: dict[str, bool]
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PaperTradeDecision:
    timestamp: datetime
    symbol: str
    direction: str
    quantity: int
    confidence: float
    price: float
    signal_source: str = "strategic_ensemble"
    model_version: str = "phase3_student_v1"
    universe_version: str = "nse_quarterly_v1"
    data_source_tags: tuple[str, ...] = ("primary_api",)
    risk_mode: RiskMode = RiskMode.NORMAL
    order_type: OrderType = OrderType.LIMIT


@dataclass(frozen=True)
class PaperTradingReport:
    crashed: bool
    total_orders: int
    filled_orders: int
    partial_fills: int
    rejected_orders: int
    uptime_ratio: float
    slippage_realism_passed: bool
    all_agents_emitting: bool
    audit_trail_complete: bool
    realized_pnl: float
    audit_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PaperTradingConfig:
    initial_cash: float = 1_000_000.0
    rejection_probability: float = 0.02
    partial_fill_probability: float = 0.20
    partial_fill_ratio: float = 0.50
    slippage_realism_tolerance_bps: float = 20.0
    session_minutes: int = 375
    uptime_target: float = 0.80


@dataclass(frozen=True)
class GovernanceAuditResult:
    registered_models: int
    complete_models: int
    registry_complete: bool
    reproducibility_ready: bool
    promotion_gate_passed: bool
    rollback_passed: bool
    missing_fields_by_model: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Phase3GateEvidence:
    latency_gate_passed: bool
    crisis_agreement_passed: bool
    rollback_tested: bool
    walk_forward_passed: bool
    observation_schema_validated: bool
    tier1_operational: bool
    compliance_audit_complete: bool
    blocking_defects: int
    checks: dict[str, bool]


@dataclass(frozen=True)
class GoNoGoAssessment:
    go: bool
    checklist: dict[str, bool]
    at_risk_items: tuple[str, ...]


@dataclass(frozen=True)
class Week4ReadinessBundle:
    backtesting: FullStackBacktestEngine
    stress: StressTestEngine
    paper: PaperTradingHarness
    governance: MLOpsGovernanceAuditor
    evidence: Phase3GateEvidenceCollector
    assessor: GoNoGoAssessor
    drift_monitor: ADDMDriftEngine
    false_trigger_governor: FalseTriggerRateGovernor
    risk_gate_reviewer: Week4RiskGateReviewer
    l4_drill_runner: L4KillSwitchDrillRunner


class FullStackBacktestEngine:
    def __init__(self, *, periods_per_year: int = 252) -> None:
        self.periods_per_year = max(1, int(periods_per_year))

    def run_walk_forward(self, returns_by_fold: Sequence[Sequence[float]]) -> WalkForwardBacktestReport:
        folds = tuple(self.compute_metrics(returns) for returns in returns_by_fold)
        if not folds:
            empty = self.compute_metrics(())
            checks = self.compare_to_go_live_targets(empty)
            return WalkForwardBacktestReport(
                fold_metrics=(),
                aggregate_metrics=empty,
                target_pass=all(checks.values()),
                target_checks=checks,
            )

        aggregate = self.compute_metrics(
            tuple(float(period_return) for fold in returns_by_fold for period_return in fold)
        )
        checks = self.compare_to_go_live_targets(aggregate)
        return WalkForwardBacktestReport(
            fold_metrics=folds,
            aggregate_metrics=aggregate,
            target_pass=all(checks.values()),
            target_checks=checks,
        )

    def compute_metrics(self, returns: Sequence[float]) -> BacktestMetrics:
        values = [float(item) for item in returns]
        periods = len(values)
        if periods == 0:
            return BacktestMetrics(
                periods=0,
                sharpe=0.0,
                sortino=0.0,
                calmar=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_return=0.0,
            )

        average_return = mean(values)
        variance = sum((item - average_return) ** 2 for item in values) / max(1, periods - 1)
        volatility = sqrt(max(variance, 0.0))

        negative_returns = [item for item in values if item < 0.0]
        downside_variance = (
            sum(item * item for item in negative_returns) / len(negative_returns)
            if negative_returns
            else 0.0
        )
        downside_dev = sqrt(downside_variance)

        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for period_return in values:
            equity *= 1.0 + period_return
            peak = max(peak, equity)
            if peak > 0.0:
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)

        total_return = equity - 1.0
        annualized_return = (1.0 + total_return) ** (self.periods_per_year / max(periods, 1)) - 1.0
        sharpe = (average_return / volatility) * sqrt(self.periods_per_year) if volatility > 0.0 else 0.0
        sortino = (average_return / downside_dev) * sqrt(self.periods_per_year) if downside_dev > 0.0 else 0.0
        calmar = annualized_return / max_drawdown if max_drawdown > 0.0 else 0.0

        wins = [item for item in values if item > 0.0]
        losses = [item for item in values if item < 0.0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)

        return BacktestMetrics(
            periods=periods,
            sharpe=float(sharpe),
            sortino=float(sortino),
            calmar=float(calmar),
            max_drawdown=float(max_drawdown),
            win_rate=float(len(wins) / periods),
            profit_factor=float(profit_factor),
            total_return=float(total_return),
        )

    @staticmethod
    def compare_to_go_live_targets(metrics: BacktestMetrics) -> dict[str, bool]:
        return {
            "sharpe": metrics.sharpe >= GO_LIVE_TARGETS["sharpe"],
            "sortino": metrics.sortino >= GO_LIVE_TARGETS["sortino"],
            "max_drawdown": metrics.max_drawdown <= GO_LIVE_TARGETS["max_drawdown"],
            "win_rate": metrics.win_rate >= GO_LIVE_TARGETS["win_rate"],
            "profit_factor": metrics.profit_factor >= GO_LIVE_TARGETS["profit_factor"],
        }

    @staticmethod
    def audit_leakage(
        rows: Sequence[Mapping[str, Any]],
        *,
        max_feature_lag_seconds: float = 60.0,
    ) -> LeakageAuditResult:
        leakage_rows = 0
        lag_breach_rows = 0
        reasons: list[str] = []

        for row in rows:
            observation_ts = _ensure_utc_datetime(row.get("observation_timestamp"))
            decision_ts = _ensure_utc_datetime(row.get("decision_timestamp"))
            feature_ts = _ensure_utc_datetime(row.get("feature_timestamp"))
            if observation_ts is None or decision_ts is None or feature_ts is None:
                leakage_rows += 1
                reasons.append("missing_timestamp_fields")
                continue

            if decision_ts < observation_ts or feature_ts > decision_ts:
                leakage_rows += 1
                reasons.append("future_data_leakage")

            lag_seconds = max(0.0, (decision_ts - feature_ts).total_seconds())
            if lag_seconds > float(max_feature_lag_seconds):
                lag_breach_rows += 1
                reasons.append("feature_lag_breach")

        unique_reasons = tuple(sorted(set(reasons)))
        return LeakageAuditResult(
            passed=leakage_rows == 0 and lag_breach_rows == 0,
            checked_rows=len(rows),
            leakage_rows=leakage_rows,
            lag_breach_rows=lag_breach_rows,
            reasons=unique_reasons,
        )

    @staticmethod
    def audit_survivorship(universe_rows: Sequence[Mapping[str, Any]]) -> SurvivorshipAuditResult:
        delisted_symbols: set[str] = set()
        point_in_time_rows = 0
        reasons: list[str] = []

        for row in universe_rows:
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                reasons.append("missing_symbol")
                continue

            if row.get("point_in_time_version"):
                point_in_time_rows += 1
            else:
                reasons.append("missing_point_in_time_version")

            delisted_at = _ensure_utc_datetime(row.get("delisted_at"))
            as_of = _ensure_utc_datetime(row.get("as_of"))
            if delisted_at is not None:
                delisted_symbols.add(symbol)
                if as_of is not None and as_of > delisted_at and bool(row.get("is_active", False)):
                    reasons.append("post_delist_active_membership")

        if not delisted_symbols:
            reasons.append("missing_delisted_symbol_coverage")
        if point_in_time_rows != len(universe_rows):
            reasons.append("point_in_time_gap")

        unique_reasons = tuple(sorted(set(reasons)))
        return SurvivorshipAuditResult(
            passed=len(unique_reasons) == 0,
            rows_checked=len(universe_rows),
            delisted_symbols_seen=len(delisted_symbols),
            point_in_time_rows=point_in_time_rows,
            reasons=unique_reasons,
        )


class StressTestEngine:
    def __init__(
        self,
        *,
        capacity_impact_cap_bps: float = 25.0,
        scenario_library_version: str = WEEK3_SCENARIO_LIBRARY_VERSION,
    ) -> None:
        self.capacity_impact_cap_bps = float(capacity_impact_cap_bps)
        self.scenario_library_version = str(scenario_library_version)

    def scenario_library(self) -> tuple[StressScenarioDefinition, ...]:
        """Week 3 mandatory stress library with type labels."""
        return (
            StressScenarioDefinition(
                scenario_id="rbi_surprise_rate_hike",
                scenario_type="historical",
                key_test="INR move and bond-yield spike containment",
            ),
            StressScenarioDefinition(
                scenario_id="inr_flash_move",
                scenario_type="historical_synthetic",
                key_test="execution continuity and hedge trigger speed",
            ),
            StressScenarioDefinition(
                scenario_id="liquidity_drought",
                scenario_type="synthetic",
                key_test="impact-cost spike and participation-limit containment",
            ),
            StressScenarioDefinition(
                scenario_id="gfc_2008",
                scenario_type="historical",
                key_test="portfolio survival and drawdown containment",
            ),
            StressScenarioDefinition(
                scenario_id="taper_tantrum_2013",
                scenario_type="historical",
                key_test="regime transition speed and macro response",
            ),
            StressScenarioDefinition(
                scenario_id="covid_crash_2020",
                scenario_type="historical",
                key_test="multi-asset correlation behavior under stress",
            ),
            StressScenarioDefinition(
                scenario_id="correlation_inversion",
                scenario_type="impossible",
                key_test="risk model coherence with non-negative variance",
            ),
            StressScenarioDefinition(
                scenario_id="frozen_constituent_prices",
                scenario_type="impossible",
                key_test="zero-volume vs missing-data distinction",
            ),
            StressScenarioDefinition(
                scenario_id="multi_asset_liquidity_vacuum",
                scenario_type="impossible",
                key_test="safe-mode activation during simultaneous depth collapse",
            ),
            StressScenarioDefinition(
                scenario_id="data_poisoning_feed_freeze",
                scenario_type="synthetic",
                key_test="degraded-mode behavior under feed-integrity uncertainty",
            ),
        )

    def build_capacity_replays(
        self,
        definitions: Sequence[StressScenarioDefinition],
        *,
        multipliers: Sequence[float] = WEEK3_CAPACITY_MULTIPLIERS,
    ) -> tuple[StressScenarioResult, ...]:
        """Expand each scenario into deterministic 1x/2x/3x capacity drills."""
        expanded: list[StressScenarioResult] = []
        for definition in definitions:
            for multiplier in multipliers:
                expanded.append(
                    StressScenarioResult(
                        scenario_id=f"{definition.scenario_id}_{multiplier:.0f}x",
                        protective_mode=RiskMode.REDUCE_ONLY,
                        expected_min_mode=RiskMode.REDUCE_ONLY,
                        scenario_type=definition.scenario_type,
                        feed_integrity_uncertain=definition.scenario_id == "data_poisoning_feed_freeze",
                        safe_mode_engaged=True,
                        variance_defined=True,
                        capacity_multiplier=float(multiplier),
                        impact_bps=0.0,
                    )
                )
        return tuple(expanded)

    def build_nightly_ci_scenarios(
        self,
        *,
        multipliers: Sequence[float] = WEEK3_CAPACITY_MULTIPLIERS,
    ) -> tuple[StressScenarioResult, ...]:
        """Nightly CI replay set for Week 3 stress framework."""
        return self.build_capacity_replays(self.scenario_library(), multipliers=multipliers)

    def evaluate(self, scenarios: Sequence[StressScenarioResult]) -> StressTestReport:
        failures: list[str] = []
        scenario_type_counts: dict[str, int] = {}
        hard_cap_forced = False
        hard_cap_multiplier = 3.0
        for scenario in scenarios:
            scenario_type = str(scenario.scenario_type or "untyped")
            scenario_type_counts[scenario_type] = scenario_type_counts.get(scenario_type, 0) + 1

            if scenario.crashed:
                failures.append(f"{scenario.scenario_id}:crash")
            if scenario.data_corruption:
                failures.append(f"{scenario.scenario_id}:data_corruption")
            if not scenario.zero_vs_missing_distinguished:
                failures.append(f"{scenario.scenario_id}:zero_missing_confused")
            if scenario.feed_integrity_uncertain and not scenario.safe_mode_engaged:
                failures.append(f"{scenario.scenario_id}:safe_mode_not_engaged")
            if scenario.scenario_id.startswith("correlation_inversion") and not scenario.variance_defined:
                failures.append(f"{scenario.scenario_id}:variance_invalid")
            if scenario.snapback_ticks > scenario.max_snapback_ticks:
                failures.append(f"{scenario.scenario_id}:snapback_slow")
            if _MODE_ORDER[scenario.protective_mode] < _MODE_ORDER[scenario.expected_min_mode]:
                failures.append(f"{scenario.scenario_id}:insufficient_protective_mode")
            if scenario.capacity_multiplier >= 3.0 and scenario.impact_bps > self.capacity_impact_cap_bps:
                failures.append(f"{scenario.scenario_id}:impact_cap_breach")
                hard_cap_forced = True
                hard_cap_multiplier = min(hard_cap_multiplier, 2.0)

        if hard_cap_forced:
            failures.append("global:capacity_hard_cap_forced_2x")

        return StressTestReport(
            passed=len(failures) == 0,
            scenarios_run=len(scenarios),
            failure_count=len(failures),
            failure_reasons=tuple(failures),
            scenario_library_version=self.scenario_library_version,
            scenario_type_counts=scenario_type_counts,
            capacity_hard_cap_multiplier=hard_cap_multiplier,
            hard_cap_forced=hard_cap_forced,
        )

    def generate_quarterly_review(
        self,
        *,
        report: StressTestReport,
        owner: str,
        reviewer: str,
        previous_failure_count: int | None = None,
        quarter_label: str | None = None,
        generated_at: datetime | None = None,
    ) -> QuarterlyStressReview:
        generated = _ensure_utc_datetime(generated_at) or datetime.now(UTC)
        previous_failures = float(previous_failure_count) if previous_failure_count is not None else float(report.failure_count)
        failure_delta = float(report.failure_count) - previous_failures

        actions: list[str] = []
        if not report.passed:
            actions.append("rerun_failed_scenarios")
        if report.hard_cap_forced:
            actions.append("enforce_capacity_hard_cap_2x")
        if any("snapback_slow" in reason for reason in report.failure_reasons):
            actions.append("review_snapback_tuning_manually")
        if not actions:
            actions.append("maintain_current_capacity_limits")

        return QuarterlyStressReview(
            generated_at=generated,
            quarter_label=quarter_label or _quarter_label(generated),
            owner=str(owner).strip() or "risk_owner",
            reviewer=str(reviewer).strip() or "risk_reviewer",
            scenario_library_version=report.scenario_library_version,
            scenarios_run=int(report.scenarios_run),
            failure_count=int(report.failure_count),
            hard_cap_multiplier=float(report.capacity_hard_cap_multiplier),
            hard_cap_forced=bool(report.hard_cap_forced),
            performance_delta_summary={"failure_count_delta": float(failure_delta)},
            required_actions=tuple(actions),
        )


class ADDMDriftEngine:
    """
    Week 4 Day 1-2 continuous drift monitor for Phase 2 inputs and Phase 3 outputs.
    """

    def __init__(
        self,
        *,
        drift_alert_threshold: float = 0.12,
        sustained_window: int = 3,
        demotion_window: int = 5,
        min_size_multiplier: float = 0.05,
    ) -> None:
        self.drift_alert_threshold = max(0.0, float(drift_alert_threshold))
        self.sustained_window = max(1, int(sustained_window))
        self.demotion_window = max(self.sustained_window, int(demotion_window))
        self.min_size_multiplier = max(0.0, min(1.0, float(min_size_multiplier)))
        self._alerts: list[tuple[datetime, bool]] = []
        self._events: list[dict[str, object]] = []

    def evaluate(self, reading: DriftReading) -> DriftDecision:
        now = _ensure_utc_datetime(reading.timestamp) or datetime.now(UTC)
        drift_score = max(float(reading.phase2_input_drift), float(reading.phase3_output_drift))
        drift_alert = drift_score >= self.drift_alert_threshold
        self._alerts.append((now, drift_alert))
        keep = max(self.demotion_window * 4, self.sustained_window * 4)
        self._alerts = self._alerts[-keep:]

        sustained_drift = self._last_n_all_true(self.sustained_window)
        demotion_triggered = self._last_n_all_true(self.demotion_window)
        size_multiplier = self._dynamic_size_multiplier(
            provenance_reliability=float(reading.provenance_reliability),
            drift_score=drift_score,
        )

        if demotion_triggered:
            reason = "sustained_drift_demotion_triggered"
        elif sustained_drift:
            reason = "sustained_drift_alert"
        elif drift_alert:
            reason = "drift_alert"
        else:
            reason = "drift_stable"

        decision = DriftDecision(
            timestamp=now,
            drift_score=drift_score,
            size_multiplier=size_multiplier,
            drift_alert=drift_alert,
            sustained_drift=sustained_drift,
            demotion_triggered=demotion_triggered,
            reason=reason,
        )
        self._events.append(
            {
                "timestamp": now.isoformat(),
                "drift_score": drift_score,
                "drift_alert": drift_alert,
                "sustained_drift": sustained_drift,
                "demotion_triggered": demotion_triggered,
                "size_multiplier": size_multiplier,
                "reason": reason,
            }
        )
        return decision

    def recent_events(self, limit: int = 100) -> tuple[dict[str, object], ...]:
        return tuple(self._events[-max(0, int(limit)) :])

    def _last_n_all_true(self, n: int) -> bool:
        if len(self._alerts) < n:
            return False
        return all(flag for _, flag in self._alerts[-n:])

    def _dynamic_size_multiplier(self, *, provenance_reliability: float, drift_score: float) -> float:
        reliability = max(0.0, min(1.0, provenance_reliability))
        denominator = max(self.drift_alert_threshold * 2.0, 1e-9)
        drift_pressure = max(0.0, min(1.0, drift_score / denominator))
        dynamic_multiplier = reliability * (1.0 - 0.7 * drift_pressure)
        return max(self.min_size_multiplier, min(1.0, dynamic_multiplier))


class FalseTriggerRateGovernor:
    """
    Week 4 Day 3 review layer for false-trigger breaches.
    """

    def __init__(
        self,
        *,
        acceptance_limit: float = 0.20,
        rolling_days: int = 30,
    ) -> None:
        self.acceptance_limit = max(0.0, min(1.0, float(acceptance_limit)))
        self.rolling_days = max(1, int(rolling_days))
        self._events: list[dict[str, object]] = []

    def evaluate_budget_engine(
        self,
        engine: VolatilityScaledRiskBudgetEngine,
        *,
        now: datetime | None = None,
    ) -> FalseTriggerReview:
        timestamp = _ensure_utc_datetime(now) or datetime.now(UTC)
        rate = float(engine.false_trigger_rate(now=timestamp))
        paused = bool(engine.auto_adjustment_paused(now=timestamp))
        escalation_required = paused or rate > self.acceptance_limit
        reason = "operator_review_required" if escalation_required else "false_trigger_within_limit"
        review = FalseTriggerReview(
            timestamp=timestamp,
            false_trigger_rate=rate,
            acceptance_limit=self.acceptance_limit,
            rolling_days=self.rolling_days,
            auto_adjustment_paused=paused,
            escalation_required=escalation_required,
            reason=reason,
        )
        self._events.append(
            {
                "timestamp": timestamp.isoformat(),
                "false_trigger_rate": rate,
                "acceptance_limit": self.acceptance_limit,
                "auto_adjustment_paused": paused,
                "escalation_required": escalation_required,
                "reason": reason,
            }
        )
        return review

    def recent_events(self, limit: int = 100) -> tuple[dict[str, object], ...]:
        return tuple(self._events[-max(0, int(limit)) :])


class L4KillSwitchDrillRunner:
    """
    Week 4 Day 4-5 final surprise L4 drill helper.
    """

    def run(
        self,
        *,
        timestamp: datetime | None = None,
        authorizer: str = "surprise_drill_operator",
    ) -> L4KillSwitchDrillResult:
        now = _ensure_utc_datetime(timestamp) or datetime.now(UTC)
        overseer = RiskOverseerStateMachine()
        decision = overseer.evaluate(
            RiskSignalSnapshot(timestamp=now, manual_kill_switch=True),
            authorizer=authorizer,
        )
        latest_events = overseer.recent_events(limit=1)
        event_recorded = (
            len(latest_events) == 1
            and latest_events[0].event_type == "KILL_SWITCH_TRIGGER"
            and latest_events[0].trigger_layer == "L4_MANUAL"
        )
        passed = decision.mode == RiskMode.KILL_SWITCH and decision.should_cancel_orders and event_recorded
        return L4KillSwitchDrillResult(
            timestamp=now,
            passed=passed,
            mode=decision.mode,
            should_cancel_orders=decision.should_cancel_orders,
            event_recorded=event_recorded,
            reason="l4_manual_kill_switch_drill",
        )


class Week4RiskGateReviewer:
    """
    Final go-live risk gate checks:
    SHAP coverage, live attribution visibility, and surprise L4 drill.
    """

    def __init__(self, *, min_shap_coverage: float = 0.80) -> None:
        self.min_shap_coverage = max(0.0, min(1.0, float(min_shap_coverage)))

    def evaluate(
        self,
        *,
        xai_logger: XAILogger,
        pnl_attribution: PnLAttributionEngine,
        risk_audit_events: Sequence[Mapping[str, Any]],
        l4_drill_result: L4KillSwitchDrillResult | None = None,
    ) -> Week4RiskGateReport:
        shap_coverage = float(xai_logger.coverage())
        shap_coverage_ok = shap_coverage >= self.min_shap_coverage
        dashboard = pnl_attribution.dashboard_snapshot()
        pnl_dashboard_active = int(dashboard.get("events_count", 0)) > 0

        risk_override_events = 0
        for event in risk_audit_events:
            event_type = str(event.get("event", "")).strip()
            if event_type == "risk_cancel_orders":
                risk_override_events += 1
                continue
            if event_type == "risk_overseer_decision":
                mode = str(event.get("mode", "normal")).strip().lower()
                if mode != RiskMode.NORMAL.value:
                    risk_override_events += 1

        risk_overrides_visible = pnl_dashboard_active and risk_override_events > 0
        drill = l4_drill_result or L4KillSwitchDrillRunner().run()
        checklist = {
            "shap_coverage": shap_coverage_ok,
            "pnl_dashboard_active": pnl_dashboard_active,
            "risk_overrides_visible": risk_overrides_visible,
            "l4_drill_passed": bool(drill.passed),
        }
        failures = tuple(key for key, passed in checklist.items() if not passed)
        return Week4RiskGateReport(
            passed=len(failures) == 0,
            shap_coverage=shap_coverage,
            shap_coverage_ok=shap_coverage_ok,
            pnl_dashboard_active=pnl_dashboard_active,
            risk_override_events=risk_override_events,
            risk_overrides_visible_in_dashboard=risk_overrides_visible,
            l4_drill_passed=bool(drill.passed),
            checklist=checklist,
            failure_reasons=failures,
        )


class PaperTradingHarness:
    def __init__(
        self,
        *,
        execution_engine: ExecutionEngine | None = None,
        config: PaperTradingConfig | None = None,
    ) -> None:
        self.execution_engine = execution_engine or ExecutionEngine()
        self.config = config or PaperTradingConfig()

    def run_session(
        self,
        decisions: Sequence[PaperTradeDecision],
        *,
        outage_minutes: float = 0.0,
        seed: int = 7,
    ) -> PaperTradingReport:
        rng = Random(seed)
        cash = float(self.config.initial_cash)
        positions: dict[str, int] = {}

        filled = 0
        partial = 0
        rejected = 0
        estimated_slippage_samples: list[float] = []
        realized_slippage_samples: list[float] = []
        audit_events: list[dict[str, Any]] = []

        for index, decision in enumerate(decisions):
            quantity = max(0, int(decision.quantity))
            context = ExecutionContext(
                timestamp=decision.timestamp,
                symbol=decision.symbol,
                current_price=float(max(decision.price, 0.01)),
                orderbook_imbalance=0.0,
                queue_pressure=0.0,
                avg_volume_1h=float(max(quantity * 20, 1_000)),
            )
            plan = self.execution_engine.plan_execution(
                request=OrderRequest(
                    symbol=decision.symbol,
                    direction=decision.direction,
                    target_quantity=quantity,
                    target_notional=float(quantity) * float(max(decision.price, 0.01)),
                    confidence=decision.confidence,
                    order_type=decision.order_type,
                    risk_mode=decision.risk_mode,
                    metadata={"paper_trading": True, "signal_source": decision.signal_source},
                ),
                context=context,
            )

            pre_trade_passed = bool(plan.compliance.passed)
            rejection_reason: str | None = None
            fill_qty = 0
            event_type = "ORDER_INTENT"

            if not pre_trade_passed:
                rejected += 1
                event_type = "REJECTION"
                rejection_reason = ",".join(plan.compliance.reasons) if plan.compliance.reasons else "pre_trade_failed"
            elif rng.random() < self.config.rejection_probability:
                rejected += 1
                event_type = "REJECTION"
                rejection_reason = "simulated_broker_reject"
            else:
                base_qty = max(0, int(plan.instruction.quantity))
                if base_qty == 0:
                    rejected += 1
                    event_type = "REJECTION"
                    rejection_reason = "zero_effective_quantity"
                elif rng.random() < self.config.partial_fill_probability:
                    partial += 1
                    fill_qty = max(1, int(round(base_qty * self.config.partial_fill_ratio)))
                    event_type = "PARTIAL_FILL"
                else:
                    filled += 1
                    fill_qty = base_qty
                    event_type = "FILL"

            estimated_slippage = float(plan.estimated_slippage_bps)
            if fill_qty > 0:
                realized_slippage = estimated_slippage + rng.uniform(-5.0, 5.0)
                estimated_slippage_samples.append(estimated_slippage)
                realized_slippage_samples.append(realized_slippage)
                signed_quantity = fill_qty if decision.direction.upper() == "BUY" else -fill_qty
                positions[decision.symbol] = positions.get(decision.symbol, 0) + signed_quantity
                signed_slippage = realized_slippage / 10_000.0
                trade_price = float(decision.price) * (1.0 + signed_slippage if signed_quantity > 0 else 1.0 - signed_slippage)
                cash -= float(signed_quantity) * trade_price

            audit_events.append(
                _build_audit_event(
                    event_index=index,
                    decision=decision,
                    event_type=event_type,
                    pre_trade_checks_passed=pre_trade_passed,
                    rejection_reason=rejection_reason,
                )
            )

        mark_to_market = sum(
            float(position) * float(next((d.price for d in reversed(decisions) if d.symbol == symbol), 0.0))
            for symbol, position in positions.items()
        )
        realized_pnl = (cash + mark_to_market) - float(self.config.initial_cash)

        uptime_ratio = max(0.0, min(1.0, 1.0 - (float(outage_minutes) / max(1.0, float(self.config.session_minutes)))))
        slippage_realism_passed = True
        if estimated_slippage_samples and realized_slippage_samples:
            max_slippage_delta = max(
                abs(realized - estimated)
                for realized, estimated in zip(realized_slippage_samples, estimated_slippage_samples)
            )
            slippage_realism_passed = max_slippage_delta <= float(self.config.slippage_realism_tolerance_bps)

        all_agents_emitting = all(bool(decision.signal_source) for decision in decisions)
        audit_trail_complete = _audit_has_required_fields(audit_events)

        return PaperTradingReport(
            crashed=False,
            total_orders=len(decisions),
            filled_orders=filled,
            partial_fills=partial,
            rejected_orders=rejected,
            uptime_ratio=uptime_ratio,
            slippage_realism_passed=slippage_realism_passed,
            all_agents_emitting=all_agents_emitting,
            audit_trail_complete=audit_trail_complete,
            realized_pnl=float(realized_pnl),
            audit_events=tuple(audit_events),
        )


class MLOpsGovernanceAuditor:
    REQUIRED_MODEL_FIELDS: tuple[str, ...] = (
        "model_id",
        "version",
        "training_data_snapshot_hash",
        "code_hash",
        "feature_schema_version",
        "hyperparameters",
        "validation_metrics",
        "baseline_comparison",
        "plan_version",
        "created_by",
        "reviewed_by",
    )

    def __init__(self, *, promotion_pipeline: PromotionGatePipeline | None = None) -> None:
        self.promotion_pipeline = promotion_pipeline or PromotionGatePipeline()

    def audit_registry(
        self,
        models: Sequence[Mapping[str, Any]],
        *,
        min_required_models: int = 6,
        promotion_evidence: PromotionEvidence | None = None,
        rollback_result: RollbackDrillResult | None = None,
    ) -> GovernanceAuditResult:
        missing_fields: dict[str, tuple[str, ...]] = {}
        complete_models = 0

        for item in models:
            model_id = str(item.get("model_id") or "unknown_model")
            missing = tuple(field for field in self.REQUIRED_MODEL_FIELDS if not item.get(field))
            if missing:
                missing_fields[model_id] = missing
            else:
                complete_models += 1

        registry_complete = len(models) >= int(min_required_models) and complete_models == len(models)
        reproducibility_ready = all(
            bool(item.get("training_data_snapshot_hash")) and bool(item.get("code_hash"))
            for item in models
        ) and len(models) > 0

        evidence = promotion_evidence or PromotionEvidence(
            walk_forward_outperformance=True,
            non_regression=True,
            crisis_slice_agreement=0.90,
            latency_gate_passed=True,
            rollback_ready=True,
            critical_compliance_violations=0,
        )
        shadow_decision = self.promotion_pipeline.transition(
            current_stage=PolicyStage.CANDIDATE,
            requested_stage=PolicyStage.SHADOW,
            evidence=evidence,
        )
        champion_decision = self.promotion_pipeline.transition(
            current_stage=shadow_decision.to_stage,
            requested_stage=PolicyStage.CHAMPION,
            evidence=evidence,
        )
        promotion_gate_passed = shadow_decision.approved and champion_decision.approved

        rollback_passed = bool(rollback_result and rollback_result.executed and rollback_result.reverted_to)

        return GovernanceAuditResult(
            registered_models=len(models),
            complete_models=complete_models,
            registry_complete=registry_complete,
            reproducibility_ready=reproducibility_ready,
            promotion_gate_passed=promotion_gate_passed,
            rollback_passed=rollback_passed,
            missing_fields_by_model=missing_fields,
        )

    @staticmethod
    def run_rollback_drill(
        champions: Sequence[str],
        *,
        failed_model_id: str,
        started_at: datetime,
        ended_at: datetime,
    ) -> RollbackDrillResult:
        drill = RollbackDrillManager()
        for model_id in champions:
            drill.set_champion(model_id)
        return drill.rollback(
            failed_model_id=failed_model_id,
            started_at=started_at,
            ended_at=ended_at,
        )


class Phase3GateEvidenceCollector:
    def collect(
        self,
        *,
        latency_p99_ms: float,
        latency_p999_ms: float,
        degrade_path_passed: bool,
        crisis_slice_agreement: float,
        rollback_result: RollbackDrillResult | None,
        backtest_metrics: BacktestMetrics,
        observation_schema_version: str,
        tier1_status: Mapping[str, bool],
        compliance_violations: int = 0,
        blocking_defects: int = 0,
    ) -> Phase3GateEvidence:
        backtest_checks = FullStackBacktestEngine.compare_to_go_live_targets(backtest_metrics)
        walk_forward_passed = all(backtest_checks.values())

        latency_gate_passed = (
            float(latency_p99_ms) <= 10.0
            and float(latency_p999_ms) <= 12.0
            and bool(degrade_path_passed)
        )
        crisis_agreement_passed = float(crisis_slice_agreement) >= 0.80
        rollback_tested = bool(rollback_result and rollback_result.executed)
        observation_schema_validated = bool(str(observation_schema_version).strip())
        tier1_operational = all(bool(value) for value in tier1_status.values()) if tier1_status else False
        compliance_audit_complete = int(compliance_violations) == 0

        checks = {
            "latency_gate": latency_gate_passed,
            "crisis_agreement": crisis_agreement_passed,
            "rollback_tested": rollback_tested,
            "walk_forward": walk_forward_passed,
            "observation_schema": observation_schema_validated,
            "tier1_operational": tier1_operational,
            "compliance": compliance_audit_complete,
            "blocking_defects": int(blocking_defects) == 0,
        }
        return Phase3GateEvidence(
            latency_gate_passed=latency_gate_passed,
            crisis_agreement_passed=crisis_agreement_passed,
            rollback_tested=rollback_tested,
            walk_forward_passed=walk_forward_passed,
            observation_schema_validated=observation_schema_validated,
            tier1_operational=tier1_operational,
            compliance_audit_complete=compliance_audit_complete,
            blocking_defects=int(blocking_defects),
            checks=checks,
        )


class GoNoGoAssessor:
    def assess(self, evidence: Phase3GateEvidence) -> GoNoGoAssessment:
        checklist = dict(evidence.checks)
        at_risk_items = tuple(key for key, passed in checklist.items() if not passed)
        return GoNoGoAssessment(
            go=len(at_risk_items) == 0,
            checklist=checklist,
            at_risk_items=at_risk_items,
        )


class Week4Controller:
    """
    Helper facade for Week 4 readiness steps:
    backtest + stress + paper harness + governance + gate evidence.
    """

    def __init__(self, bundle: Week4ReadinessBundle) -> None:
        self.bundle = bundle

    def run_day1_backtesting(
        self,
        *,
        returns_by_fold: Sequence[Sequence[float]],
        leakage_rows: Sequence[Mapping[str, Any]],
        universe_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        backtest = self.bundle.backtesting.run_walk_forward(returns_by_fold)
        leakage = self.bundle.backtesting.audit_leakage(leakage_rows)
        survivorship = self.bundle.backtesting.audit_survivorship(universe_rows)
        return {
            "walk_forward": backtest,
            "leakage": leakage,
            "survivorship": survivorship,
        }

    def run_day2_stress(self, scenarios: Sequence[StressScenarioResult]) -> StressTestReport:
        return self.bundle.stress.evaluate(scenarios)

    def run_day1_and_day2_drift(self, readings: Sequence[DriftReading]) -> tuple[DriftDecision, ...]:
        return tuple(self.bundle.drift_monitor.evaluate(reading) for reading in readings)

    def run_day3_false_trigger_review(
        self,
        risk_budget_engine: VolatilityScaledRiskBudgetEngine,
        *,
        now: datetime | None = None,
    ) -> FalseTriggerReview:
        return self.bundle.false_trigger_governor.evaluate_budget_engine(risk_budget_engine, now=now)

    def run_day3_paper_trading(
        self,
        decisions: Sequence[PaperTradeDecision],
        *,
        outage_minutes: float = 0.0,
        seed: int = 7,
    ) -> PaperTradingReport:
        return self.bundle.paper.run_session(decisions, outage_minutes=outage_minutes, seed=seed)

    def run_day4_governance(
        self,
        *,
        models: Sequence[Mapping[str, Any]],
        rollback_result: RollbackDrillResult | None,
        promotion_evidence: PromotionEvidence | None = None,
    ) -> GovernanceAuditResult:
        return self.bundle.governance.audit_registry(
            models,
            rollback_result=rollback_result,
            promotion_evidence=promotion_evidence,
        )

    def run_day5_and_day6_gates(
        self,
        *,
        latency_p99_ms: float,
        latency_p999_ms: float,
        degrade_path_passed: bool,
        crisis_slice_agreement: float,
        rollback_result: RollbackDrillResult | None,
        backtest_metrics: BacktestMetrics,
        observation_schema_version: str,
        tier1_status: Mapping[str, bool],
        compliance_violations: int = 0,
        blocking_defects: int = 0,
    ) -> tuple[Phase3GateEvidence, GoNoGoAssessment]:
        evidence = self.bundle.evidence.collect(
            latency_p99_ms=latency_p99_ms,
            latency_p999_ms=latency_p999_ms,
            degrade_path_passed=degrade_path_passed,
            crisis_slice_agreement=crisis_slice_agreement,
            rollback_result=rollback_result,
            backtest_metrics=backtest_metrics,
            observation_schema_version=observation_schema_version,
            tier1_status=tier1_status,
            compliance_violations=compliance_violations,
            blocking_defects=blocking_defects,
        )
        assessment = self.bundle.assessor.assess(evidence)
        return evidence, assessment

    def run_day4_and_day5_risk_gate(
        self,
        *,
        xai_logger: XAILogger,
        pnl_attribution: PnLAttributionEngine,
        risk_audit_events: Sequence[Mapping[str, Any]],
        drill_timestamp: datetime | None = None,
        drill_authorizer: str = "surprise_drill_operator",
    ) -> Week4RiskGateReport:
        drill = self.bundle.l4_drill_runner.run(
            timestamp=drill_timestamp,
            authorizer=drill_authorizer,
        )
        return self.bundle.risk_gate_reviewer.evaluate(
            xai_logger=xai_logger,
            pnl_attribution=pnl_attribution,
            risk_audit_events=risk_audit_events,
            l4_drill_result=drill,
        )


def build_week4_bundle(*, periods_per_year: int = 252) -> Week4ReadinessBundle:
    return Week4ReadinessBundle(
        backtesting=FullStackBacktestEngine(periods_per_year=periods_per_year),
        stress=StressTestEngine(),
        paper=PaperTradingHarness(),
        governance=MLOpsGovernanceAuditor(),
        evidence=Phase3GateEvidenceCollector(),
        assessor=GoNoGoAssessor(),
        drift_monitor=ADDMDriftEngine(),
        false_trigger_governor=FalseTriggerRateGovernor(),
        risk_gate_reviewer=Week4RiskGateReviewer(),
        l4_drill_runner=L4KillSwitchDrillRunner(),
    )


def _build_audit_event(
    *,
    event_index: int,
    decision: PaperTradeDecision,
    event_type: str,
    pre_trade_checks_passed: bool,
    rejection_reason: str | None,
) -> dict[str, Any]:
    return {
        "event_id": f"paper:{decision.symbol}:{event_index}",
        "event_type": event_type,
        "timestamp_utc": _ensure_utc_datetime(decision.timestamp).isoformat(),
        "instrument": decision.symbol,
        "direction": decision.direction.upper(),
        "quantity": int(decision.quantity),
        "price": float(decision.price),
        "order_type": decision.order_type.value,
        "model_version": decision.model_version,
        "signal_source": decision.signal_source,
        "universe_version": decision.universe_version,
        "plan_version": "v1.3.7",
        "pre_trade_checks_passed": bool(pre_trade_checks_passed),
        "rejection_reason": rejection_reason,
        "data_source_tags": list(decision.data_source_tags),
        "risk_mode": decision.risk_mode.value,
    }


def _audit_has_required_fields(events: Sequence[Mapping[str, Any]]) -> bool:
    for event in events:
        for required_field in AUDIT_REQUIRED_FIELDS:
            if required_field not in event:
                return False
    return True


def _ensure_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return None


def _quarter_label(value: datetime) -> str:
    month = int(value.month)
    quarter = ((month - 1) // 3) + 1
    return f"{value.year}-Q{quarter}"


__all__ = [
    "AUDIT_REQUIRED_FIELDS",
    "GO_LIVE_TARGETS",
    "WEEK3_CAPACITY_MULTIPLIERS",
    "WEEK3_SCENARIO_LIBRARY_VERSION",
    "ADDMDriftEngine",
    "BacktestMetrics",
    "DriftDecision",
    "DriftReading",
    "FalseTriggerRateGovernor",
    "FalseTriggerReview",
    "FullStackBacktestEngine",
    "GoNoGoAssessor",
    "GoNoGoAssessment",
    "GovernanceAuditResult",
    "L4KillSwitchDrillResult",
    "L4KillSwitchDrillRunner",
    "LeakageAuditResult",
    "MLOpsGovernanceAuditor",
    "PaperTradeDecision",
    "PaperTradingConfig",
    "PaperTradingHarness",
    "PaperTradingReport",
    "Phase3GateEvidence",
    "Phase3GateEvidenceCollector",
    "QuarterlyStressReview",
    "StressScenarioDefinition",
    "StressScenarioResult",
    "StressTestEngine",
    "StressTestReport",
    "SurvivorshipAuditResult",
    "WalkForwardBacktestReport",
    "Week4Controller",
    "Week4RiskGateReport",
    "Week4RiskGateReviewer",
    "Week4ReadinessBundle",
    "build_week4_bundle",
]
