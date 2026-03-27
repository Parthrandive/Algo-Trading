from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.config import ImpactMonitorConfig
from src.agents.strategic.impact_monitor import FillEvent, ImpactSlippageMonitor, InstrumentBucket


def _fill(*, ts: datetime, realized: float, model: float = 10.0, qty: int = 1_000, adv: float = 100_000.0) -> FillEvent:
    return FillEvent(
        symbol="RELIANCE.NS",
        bucket=InstrumentBucket.LIQUID_LARGE_CAP,
        quantity=qty,
        adv=adv,
        model_slippage_bps=model,
        realized_slippage_bps=realized,
        timestamp=ts,
    )


def test_impact_breach_triggers_auto_reduction_and_override():
    monitor = ImpactSlippageMonitor(config=ImpactMonitorConfig(cooldown_seconds=60, reduction_step_fraction=0.25))
    now = datetime(2026, 4, 14, 9, 30, tzinfo=UTC)

    decision = monitor.evaluate_fill(_fill(ts=now, realized=45.0))
    assert decision.breach is True
    assert decision.event_type == "IMPACT_BREACH"
    assert decision.size_multiplier == 0.75
    assert decision.risk_override == "reduce_only"
    assert "slippage_delta_breach" in decision.reasons


def test_cooldown_and_hysteresis_restore_size_stepwise():
    monitor = ImpactSlippageMonitor(
        config=ImpactMonitorConfig(cooldown_seconds=1, hysteresis_clean_fills=2, reduction_step_fraction=0.25),
    )
    t0 = datetime(2026, 4, 14, 9, 30, tzinfo=UTC)

    _ = monitor.evaluate_fill(_fill(ts=t0, realized=45.0))
    assert monitor.size_multiplier("RELIANCE.NS") == 0.75

    # During cooldown, clean fills do not increment recovery streak.
    _ = monitor.evaluate_fill(_fill(ts=t0 + timedelta(milliseconds=500), realized=12.0))
    assert monitor.size_multiplier("RELIANCE.NS") == 0.75

    # After cooldown, two clean fills restore by one configured step (0.25).
    _ = monitor.evaluate_fill(_fill(ts=t0 + timedelta(seconds=2), realized=12.0))
    restored = monitor.evaluate_fill(_fill(ts=t0 + timedelta(seconds=3), realized=11.0))
    assert restored.event_type == "IMPACT_RESTORE_STEP"
    assert monitor.size_multiplier("RELIANCE.NS") == 1.0
    assert restored.risk_override is None


def test_dashboard_snapshot_contains_bucket_metrics():
    monitor = ImpactSlippageMonitor()
    now = datetime(2026, 4, 14, 9, 30, tzinfo=UTC)
    monitor.evaluate_fill(_fill(ts=now, realized=35.0))
    monitor.evaluate_fill(_fill(ts=now + timedelta(seconds=10), realized=12.0))

    snapshot = monitor.dashboard_snapshot()
    assert "bucket_metrics" in snapshot
    assert isinstance(snapshot["bucket_metrics"], list)
    rows = [row for row in snapshot["bucket_metrics"] if row["bucket"] == "liquid_large_cap"]
    assert rows and rows[0]["fills"] >= 2
