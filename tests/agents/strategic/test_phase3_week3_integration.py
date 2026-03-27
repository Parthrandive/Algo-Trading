from __future__ import annotations

from datetime import UTC, datetime

from src.agents.strategic.execution import ExecutionContext, ExecutionEngine, OrderRequest, OrderType
from src.agents.strategic.impact_monitor import FillEvent, InstrumentBucket
from src.agents.strategic.orderbook_features import OrderBookSnapshot
from src.agents.strategic.portfolio import PortfolioOptimizer
from src.agents.strategic.risk_budgets import VolatilityReading
from src.agents.strategic.schemas import ActionType, EnsembleDecision, PolicyWeight, PortfolioIntent
from src.agents.strategic.week3 import Week3Controller, build_week3_bundle


def test_week3_controller_wires_tier1_components_and_checkpoint_report():
    bundle = build_week3_bundle({"nse_large_cap": 0.02})
    controller = Week3Controller(bundle)
    now = datetime(2026, 4, 19, 9, 30, tzinfo=UTC)

    risk_decision = controller.on_volatility(
        VolatilityReading(
            symbol="RELIANCE.NS",
            asset_cluster="nse_large_cap",
            realized_vol=0.03,
            timestamp=now,
        ),
    )
    assert risk_decision.cap_fraction == 0.70

    _ = controller.on_fill(
        FillEvent(
            symbol="RELIANCE.NS",
            bucket=InstrumentBucket.LIQUID_LARGE_CAP,
            quantity=20_000,
            adv=100_000,
            model_slippage_bps=10.0,
            realized_slippage_bps=40.0,
            timestamp=now,
        ),
    )

    engine = ExecutionEngine(impact_monitor=bundle.impact_monitor)
    plan = engine.plan_execution(
        OrderRequest(
            symbol="RELIANCE.NS",
            direction="BUY",
            target_quantity=100,
            target_notional=250_000.0,
            confidence=0.9,
            order_type=OrderType.LIMIT,
        ),
        ExecutionContext(
            timestamp=now,
            symbol="RELIANCE.NS",
            current_price=2500.0,
            orderbook_imbalance=0.1,
            queue_pressure=0.05,
            avg_volume_1h=500_000.0,
        ),
    )
    assert plan.instruction.quantity == 75

    features = controller.on_orderbook(
        OrderBookSnapshot(
            symbol="RELIANCE.NS",
            timestamp=now,
            bid_levels=(100.0, 90.0, 80.0, 70.0, 60.0),
            ask_levels=(90.0, 80.0, 70.0, 60.0, 50.0),
            bid_arrival_rate=90.0,
            ask_arrival_rate=60.0,
        ),
    )
    assert features.schema_version == "2.0"

    report = controller.tier1_checkpoint_report()
    assert report["impact_monitor_functional"] is True
    assert report["risk_budgets_functional"] is True
    assert report["orderbook_imbalance_integrated"] is True
    assert report["latency_ci_gate_ready"] is True


def test_portfolio_optimizer_applies_dynamic_exposure_cap_fraction():
    decision = EnsembleDecision(
        timestamp=datetime(2026, 4, 19, 9, 30, tzinfo=UTC),
        symbol="RELIANCE.NS",
        observation_snapshot_id="snap-1",
        action=ActionType.BUY,
        action_size=0.8,
        confidence=0.8,
        dominant_policy_id="phase3_student_v1",
        policy_weights=(PolicyWeight(policy_id="phase3_student_v1", weight=1.0, confidence=0.8, diversity_score=0.1),),
    )
    intent = PortfolioIntent(
        symbol="RELIANCE.NS",
        decision=decision,
        target_notional=300_000.0,
        target_quantity=3_000.0,
        metadata={"lot_size": 1},
    )
    result = PortfolioOptimizer().validate_intent(
        intent=intent,
        equity=1_000_000.0,
        current_price=100.0,
        exposure_cap_fraction=0.40,
    )
    assert result.approved is True
    assert result.adjusted_notional <= 60_000.0
    assert "exposure_capped:300,000>60,000" in result.warnings
