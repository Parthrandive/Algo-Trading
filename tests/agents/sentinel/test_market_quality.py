from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.sentinel.config import SessionRules
from src.agents.sentinel.market_quality import HistoricalQualityThresholds, compute_symbol_quality


def _frame(symbol: str, volumes: list[int]) -> pd.DataFrame:
    ist = ZoneInfo("Asia/Kolkata")
    timestamps = [
        datetime(2026, 2, 16, 9, 15, tzinfo=ist),
        datetime(2026, 2, 16, 10, 15, tzinfo=ist),
        datetime(2026, 2, 16, 11, 15, tzinfo=ist),
        datetime(2026, 2, 16, 12, 15, tzinfo=ist),
        datetime(2026, 2, 16, 13, 15, tzinfo=ist),
    ]
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(timestamps),
            "timestamp": timestamps,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": volumes,
        }
    )


def test_equity_zero_volume_ratio_can_fail_train_ready():
    thresholds = HistoricalQualityThresholds(
        min_rows_by_interval={"1h": 5},
        min_history_days=1,
        min_coverage_pct=50.0,
        max_zero_volume_ratio=0.2,
    )
    quality = compute_symbol_quality(
        frame=_frame("RELIANCE.NS", [0, 0, 0, 0, 0]),
        symbol="RELIANCE.NS",
        interval="1h",
        session_rules=SessionRules(),
        thresholds=thresholds,
    )

    assert quality["train_ready"] is False
    assert "high_zero_volume_ratio" in quality["quality_flags"]


def test_forex_zero_volume_ratio_is_not_treated_as_error():
    thresholds = HistoricalQualityThresholds(
        min_rows_by_interval={"1h": 5},
        min_history_days=1,
        min_coverage_pct=50.0,
        max_zero_volume_ratio=0.0,
    )
    quality = compute_symbol_quality(
        frame=_frame("USDINR=X", [0, 0, 0, 0, 0]),
        symbol="USDINR=X",
        interval="1h",
        session_rules=SessionRules(),
        thresholds=thresholds,
    )

    assert "high_zero_volume_ratio" not in quality["quality_flags"]

