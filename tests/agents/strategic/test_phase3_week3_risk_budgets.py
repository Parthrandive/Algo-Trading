from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.risk_budgets import (
    VolatilityReading,
    VolatilityRegime,
    VolatilityScaledRiskBudgetEngine,
)


def test_volatility_regimes_map_to_expected_caps():
    engine = VolatilityScaledRiskBudgetEngine({"nse_large_cap": 0.02})
    t0 = datetime(2026, 4, 15, 9, 30, tzinfo=UTC)

    normal = engine.update(
        VolatilityReading(symbol="RELIANCE.NS", asset_cluster="nse_large_cap", realized_vol=0.02, timestamp=t0),
    )
    assert normal.regime == VolatilityRegime.NORMAL
    assert normal.cap_fraction == 1.0

    elevated = engine.update(
        VolatilityReading(symbol="RELIANCE.NS", asset_cluster="nse_large_cap", realized_vol=0.03, timestamp=t0),
    )
    assert elevated.regime == VolatilityRegime.ELEVATED
    assert elevated.cap_fraction == 0.70
    assert elevated.changed is True

    extreme = engine.update(
        VolatilityReading(symbol="RELIANCE.NS", asset_cluster="nse_large_cap", realized_vol=0.05, timestamp=t0),
    )
    assert extreme.regime == VolatilityRegime.EXTREME
    assert extreme.cap_fraction == 0.15


def test_false_trigger_rate_pauses_auto_adjustment():
    engine = VolatilityScaledRiskBudgetEngine({"nse_large_cap": 0.02})
    now = datetime(2026, 4, 15, 9, 30, tzinfo=UTC)
    for idx in range(5):
        engine.register_trigger_outcome(timestamp=now + timedelta(minutes=idx), was_false_trigger=True)
    engine.register_trigger_outcome(timestamp=now + timedelta(minutes=6), was_false_trigger=False)

    decision = engine.update(
        VolatilityReading(
            symbol="TCS.NS",
            asset_cluster="nse_large_cap",
            realized_vol=0.06,
            timestamp=now + timedelta(minutes=7),
        ),
    )
    assert decision.auto_adjustment_paused is True
    assert decision.event_type == "RISK_CAP_PAUSED"
    assert decision.regime == VolatilityRegime.NORMAL
    assert decision.cap_fraction == 1.0
