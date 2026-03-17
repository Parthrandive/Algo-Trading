from datetime import date

from src.agents.macro.client import DateRange
from src.agents.macro.clients.nse_fiidii_client import NSEDIIFIIClient
from src.schemas.macro_data import MacroIndicatorType, SourceType


def _write_csv(path):
    path.write_text(
        "Year,All India Equity - Net\n"
        "2000-2001,-2003.4\n"
        "2001-2002,111881.3\n",
        encoding="utf-8",
    )


def test_fii_annual_csv_fallback_loads_historical_rows(tmp_path):
    csv_path = tmp_path / "FII Turnover.csv"
    _write_csv(csv_path)

    client = NSEDIIFIIClient(raw_fetcher=lambda _: {}, fii_turnover_csv_path=csv_path)
    rows = client.get_indicator(
        MacroIndicatorType.FII_FLOW,
        DateRange(start=date(2000, 1, 1), end=date(2002, 12, 31)),
    )

    assert len(rows) == 2
    assert rows[0].timestamp.date().isoformat() == "2001-03-31"
    assert rows[0].value == -2003.4
    assert rows[0].period == "Annual"
    assert rows[0].source_type == SourceType.MANUAL_OVERRIDE


def test_dii_does_not_use_fii_annual_csv_fallback(tmp_path):
    csv_path = tmp_path / "FII Turnover.csv"
    _write_csv(csv_path)

    client = NSEDIIFIIClient(raw_fetcher=lambda _: {}, fii_turnover_csv_path=csv_path)
    rows = client.get_indicator(
        MacroIndicatorType.DII_FLOW,
        DateRange(start=date(2000, 1, 1), end=date(2002, 12, 31)),
    )

    assert rows == []


def test_live_daily_row_overrides_annual_value_on_same_timestamp(tmp_path):
    csv_path = tmp_path / "FII Turnover.csv"
    _write_csv(csv_path)

    client = NSEDIIFIIClient(
        raw_fetcher=lambda _: {"date": "2001-03-31", "fii_flow": 123.0},
        fii_turnover_csv_path=csv_path,
    )
    rows = client.get_indicator(
        MacroIndicatorType.FII_FLOW,
        DateRange(start=date(2001, 1, 1), end=date(2001, 12, 31)),
    )

    assert len(rows) == 1
    assert rows[0].timestamp.date().isoformat() == "2001-03-31"
    assert rows[0].value == 123.0
    assert rows[0].period == "Daily"
    assert rows[0].source_type == SourceType.OFFICIAL_API
