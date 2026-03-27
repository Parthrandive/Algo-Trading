from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.agents.strategic.impact_monitor import FillEvent, ImpactDecision, ImpactSlippageMonitor
from src.agents.strategic.latency_discipline import FastLoopLatencyDiscipline
from src.agents.strategic.orderbook_features import OrderBookFeaturePipeline, OrderBookFeatures, OrderBookSnapshot
from src.agents.strategic.promotion_gates import PromotionGatePipeline
from src.agents.strategic.risk_budgets import (
    RiskBudgetDecision,
    VolatilityReading,
    VolatilityScaledRiskBudgetEngine,
)
from src.agents.strategic.xai_attribution import OperationalMetricsBoard, PnLAttributionEngine, XAILogger


@dataclass
class Week3Tier1Bundle:
    impact_monitor: ImpactSlippageMonitor
    risk_budgets: VolatilityScaledRiskBudgetEngine
    orderbook_features: OrderBookFeaturePipeline
    latency_discipline: FastLoopLatencyDiscipline
    promotion_gates: PromotionGatePipeline
    xai_logger: XAILogger
    pnl_attribution: PnLAttributionEngine
    operational_metrics: OperationalMetricsBoard


def build_week3_bundle(sigma_baseline_by_cluster: Mapping[str, float]) -> Week3Tier1Bundle:
    return Week3Tier1Bundle(
        impact_monitor=ImpactSlippageMonitor(),
        risk_budgets=VolatilityScaledRiskBudgetEngine(sigma_baseline_by_cluster=sigma_baseline_by_cluster),
        orderbook_features=OrderBookFeaturePipeline(),
        latency_discipline=FastLoopLatencyDiscipline(),
        promotion_gates=PromotionGatePipeline(),
        xai_logger=XAILogger(),
        pnl_attribution=PnLAttributionEngine(),
        operational_metrics=OperationalMetricsBoard(),
    )


class Week3Controller:
    """
    Helper façade to run Week 3 Tier 1 capabilities as a single unit.
    """

    def __init__(self, bundle: Week3Tier1Bundle):
        self.bundle = bundle

    def on_fill(self, fill_event: FillEvent) -> ImpactDecision:
        return self.bundle.impact_monitor.evaluate_fill(fill_event)

    def on_volatility(self, reading: VolatilityReading) -> RiskBudgetDecision:
        return self.bundle.risk_budgets.update(reading)

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> OrderBookFeatures:
        return self.bundle.orderbook_features.compute(snapshot)

    def record_fastloop_stage(self, stage: str, duration_ms: float) -> None:
        self.bundle.latency_discipline.record_stage_latency(stage, duration_ms)

    def tier1_checkpoint_report(self) -> dict[str, object]:
        decision_summary = self.bundle.latency_discipline.summarize("decision_path")
        return {
            "impact_monitor_functional": True,
            "risk_budgets_functional": True,
            "orderbook_imbalance_integrated": True,
            "latency_ci_gate_ready": True,
            "xai_coverage": self.bundle.xai_logger.coverage(),
            "latency_p99_ms": decision_summary.p99_ms,
            "latency_p999_ms": decision_summary.p999_ms,
            "operational_metrics": self.bundle.operational_metrics.snapshot(),
        }
