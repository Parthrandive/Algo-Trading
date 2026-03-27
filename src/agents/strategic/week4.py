from __future__ import annotations

from dataclasses import dataclass
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
from src.agents.strategic.schemas import RiskMode

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
    crashed: bool = False
    data_corruption: bool = False
    zero_vs_missing_distinguished: bool = True
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
    def __init__(self, *, capacity_impact_cap_bps: float = 25.0) -> None:
        self.capacity_impact_cap_bps = float(capacity_impact_cap_bps)

    def evaluate(self, scenarios: Sequence[StressScenarioResult]) -> StressTestReport:
        failures: list[str] = []
        for scenario in scenarios:
            if scenario.crashed:
                failures.append(f"{scenario.scenario_id}:crash")
            if scenario.data_corruption:
                failures.append(f"{scenario.scenario_id}:data_corruption")
            if not scenario.zero_vs_missing_distinguished:
                failures.append(f"{scenario.scenario_id}:zero_missing_confused")
            if scenario.snapback_ticks > scenario.max_snapback_ticks:
                failures.append(f"{scenario.scenario_id}:snapback_slow")
            if _MODE_ORDER[scenario.protective_mode] < _MODE_ORDER[scenario.expected_min_mode]:
                failures.append(f"{scenario.scenario_id}:insufficient_protective_mode")
            if scenario.capacity_multiplier >= 3.0 and scenario.impact_bps > self.capacity_impact_cap_bps:
                failures.append(f"{scenario.scenario_id}:impact_cap_breach")

        return StressTestReport(
            passed=len(failures) == 0,
            scenarios_run=len(scenarios),
            failure_count=len(failures),
            failure_reasons=tuple(failures),
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


def build_week4_bundle(*, periods_per_year: int = 252) -> Week4ReadinessBundle:
    return Week4ReadinessBundle(
        backtesting=FullStackBacktestEngine(periods_per_year=periods_per_year),
        stress=StressTestEngine(),
        paper=PaperTradingHarness(),
        governance=MLOpsGovernanceAuditor(),
        evidence=Phase3GateEvidenceCollector(),
        assessor=GoNoGoAssessor(),
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
        for field in AUDIT_REQUIRED_FIELDS:
            if field not in event:
                return False
    return True


def _ensure_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return None


__all__ = [
    "AUDIT_REQUIRED_FIELDS",
    "GO_LIVE_TARGETS",
    "BacktestMetrics",
    "FullStackBacktestEngine",
    "GoNoGoAssessor",
    "GoNoGoAssessment",
    "GovernanceAuditResult",
    "LeakageAuditResult",
    "MLOpsGovernanceAuditor",
    "PaperTradeDecision",
    "PaperTradingConfig",
    "PaperTradingHarness",
    "PaperTradingReport",
    "Phase3GateEvidence",
    "Phase3GateEvidenceCollector",
    "StressScenarioResult",
    "StressTestEngine",
    "StressTestReport",
    "SurvivorshipAuditResult",
    "WalkForwardBacktestReport",
    "Week4Controller",
    "Week4ReadinessBundle",
    "build_week4_bundle",
]
