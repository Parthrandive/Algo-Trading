from datetime import date, timedelta

from src.agents.macro.run_real_pipeline import (
    _INITIAL_BACKFILL_START,
    _LOOKBACK_DAYS,
    _build_indicator_range,
)
from src.schemas.macro_data import MacroIndicatorType


def test_build_indicator_range_forces_full_backfill_when_no_rows() -> None:
    now_date = date(2026, 3, 16)
    plan = _build_indicator_range(
        indicator=MacroIndicatorType.CPI,
        now_date=now_date,
        latest_date=None,
        row_count=0,
    )

    assert plan.force_full_backfill is True
    assert plan.date_range.start == _INITIAL_BACKFILL_START[MacroIndicatorType.CPI]
    assert plan.date_range.end == now_date


def test_build_indicator_range_forces_full_backfill_when_history_is_sparse() -> None:
    now_date = date(2026, 3, 16)
    plan = _build_indicator_range(
        indicator=MacroIndicatorType.FII_FLOW,
        now_date=now_date,
        latest_date=date(2026, 3, 15),
        row_count=6,
    )

    assert plan.force_full_backfill is True
    assert plan.date_range.start == _INITIAL_BACKFILL_START[MacroIndicatorType.FII_FLOW]


def test_build_indicator_range_uses_incremental_window_when_history_is_sufficient() -> None:
    now_date = date(2026, 3, 16)
    latest_date = date(2026, 3, 15)
    plan = _build_indicator_range(
        indicator=MacroIndicatorType.FII_FLOW,
        now_date=now_date,
        latest_date=latest_date,
        row_count=400,
    )

    expected_start = max(
        _INITIAL_BACKFILL_START[MacroIndicatorType.FII_FLOW],
        latest_date - timedelta(days=_LOOKBACK_DAYS[MacroIndicatorType.FII_FLOW]),
    )

    assert plan.force_full_backfill is False
    assert plan.date_range.start == expected_start
    assert plan.date_range.end == now_date


def test_build_indicator_range_caps_start_at_now_if_latest_is_future() -> None:
    now_date = date(2026, 3, 16)
    plan = _build_indicator_range(
        indicator=MacroIndicatorType.US_10Y,
        now_date=now_date,
        latest_date=date(2026, 4, 1),
        row_count=5000,
    )

    assert plan.force_full_backfill is False
    assert plan.date_range.start == now_date
    assert plan.date_range.end == now_date
