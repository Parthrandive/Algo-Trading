from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.agents.macro.client import DateRange
from src.agents.macro.clients.akshare_client import AkShareClient
from src.schemas.macro_data import MacroIndicatorType, QualityFlag, SourceType


def test_akshare_supported_indicators():
    client = AkShareClient()
    assert client.supported_indicators == frozenset(
        [MacroIndicatorType.REPO_RATE, MacroIndicatorType.US_10Y]
    )


def test_akshare_rejects_unsupported_indicator():
    client = AkShareClient()
    dr = DateRange(start=date(2026, 1, 1), end=date(2026, 1, 31))
    with pytest.raises(ValueError, match="AkShareClient does not support indicator"):
        client.get_indicator(MacroIndicatorType.CPI, dr)


def test_akshare_repo_rate_parsing_and_date_filtering():
    def repo_fetcher(_: DateRange) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "日期": ["2025-12-06", "2026-01-10", "2026-02-08"],
                "今值": ["6.00", "6.25", "6.50"],
            }
        )

    client = AkShareClient(raw_fetchers={MacroIndicatorType.REPO_RATE: repo_fetcher})
    dr = DateRange(start=date(2026, 1, 1), end=date(2026, 2, 28))

    records = client.get_indicator(MacroIndicatorType.REPO_RATE, dr)
    assert len(records) == 2
    assert [record.value for record in records] == [6.25, 6.5]
    assert records[0].timestamp == datetime(2026, 1, 10, tzinfo=UTC)
    assert records[1].timestamp == datetime(2026, 2, 8, tzinfo=UTC)

    for record in records:
        assert record.indicator_name == MacroIndicatorType.REPO_RATE
        assert record.unit == "%"
        assert record.period == "Monthly"
        assert record.source_type == SourceType.OFFICIAL_API
        assert record.quality_status in {QualityFlag.PASS, QualityFlag.WARN}
        assert record.schema_version == "1.1"


def test_akshare_us10y_mapping_and_numeric_cleanup():
    def us_fetcher(_: DateRange) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "日期": ["2026-03-01", "2026-03-02"],
                "美国国债收益率10年": ["4.11%", "4.09%"],
            }
        )

    client = AkShareClient(raw_fetchers={MacroIndicatorType.US_10Y: us_fetcher})
    dr = DateRange(start=date(2026, 3, 1), end=date(2026, 3, 2))

    records = client.get_indicator(MacroIndicatorType.US_10Y, dr)
    assert len(records) == 2
    assert [record.value for record in records] == [4.11, 4.09]
    assert all(record.indicator_name == MacroIndicatorType.US_10Y for record in records)
    assert all(record.unit == "%" for record in records)
    assert all(record.period == "Daily" for record in records)


def test_akshare_empty_payload_returns_empty_records():
    def empty_fetcher(_: DateRange) -> pd.DataFrame:
        return pd.DataFrame(columns=["日期", "今值"])

    client = AkShareClient(raw_fetchers={MacroIndicatorType.REPO_RATE: empty_fetcher})
    dr = DateRange(start=date(2026, 1, 1), end=date(2026, 1, 31))
    assert client.get_indicator(MacroIndicatorType.REPO_RATE, dr) == []
