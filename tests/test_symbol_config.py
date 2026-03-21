from __future__ import annotations

import pandas as pd

from config.symbols import (
    EQUITY_SYMBOLS,
    FOREX_SYMBOLS,
    MAX_ZERO_PCT,
    MIN_ROWS,
    SplitCounts,
    dedupe_symbols,
    is_equity,
    is_forex,
    print_symbol_selection_summary,
    validate_equity_symbol,
)


def _frame(rows: int = MIN_ROWS + 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "close": [100.0 + idx for idx in range(rows)],
        }
    )


def test_static_symbol_config_contract():
    assert FOREX_SYMBOLS == ["USDINR=X"]
    assert EQUITY_SYMBOLS == []
    assert is_forex("USDINR=X") is True
    assert is_equity("RELIANCE.NS") is True


def test_dedupe_symbols_preserves_order():
    assert dedupe_symbols(["RELIANCE.NS", "USDINR=X", "RELIANCE.NS", "TCS.NS"]) == [
        "RELIANCE.NS",
        "USDINR=X",
        "TCS.NS",
    ]


def test_validate_equity_symbol_accepts_clean_equity_frame():
    outcome = validate_equity_symbol(
        symbol="RELIANCE.NS",
        frame=_frame(),
        interval="1d",
        split_counts=SplitCounts(train_rows=220, val_rows=60, test_rows=60),
    )
    assert outcome.is_active is True
    assert outcome.reason is None


def test_validate_equity_symbol_rejects_forex_target():
    outcome = validate_equity_symbol(
        symbol="USDINR=X",
        frame=_frame(),
        interval="1d",
        split_counts=SplitCounts(train_rows=220, val_rows=60, test_rows=60),
    )
    assert outcome.is_active is False
    assert outcome.reason == "forex_context_only"


def test_validate_equity_symbol_rejects_zero_close_heavy_frame():
    frame = _frame()
    zero_rows = int(len(frame) * (MAX_ZERO_PCT + 0.10))
    frame.loc[:zero_rows, "close"] = 0.0
    outcome = validate_equity_symbol(
        symbol="RELIANCE.NS",
        frame=frame,
        interval="1d",
        split_counts=SplitCounts(train_rows=220, val_rows=60, test_rows=60),
    )
    assert outcome.is_active is False
    assert "zero_close_pct" in str(outcome.reason)


def test_print_symbol_selection_summary(capsys):
    print_symbol_selection_summary(
        active_symbols=["INFY.NS", "TCS.NS"],
        skipped_reasons={"USDINR=X": "forex_context_only"},
    )
    output = capsys.readouterr().out
    assert "Active equity symbols this run: [INFY.NS, TCS.NS]" in output
    assert "Skipped symbols: [USDINR=X (forex_context_only)]" in output
    assert "Forex (external feature): [USDINR=X]" in output
