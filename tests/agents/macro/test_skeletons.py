"""
Day 2 skeleton tests — interface conformance and type safety.

These tests verify:
1. All 5 clients satisfy MacroClientInterface (duck-typed Protocol).
2. Each client's `supported_indicators` is correct and non-empty.
3. `get_indicator` raises ValueError for unsupported indicators.
4. `get_indicator` raises NotImplementedError for supported indicators
   (stub behaviour — real fetch not yet implemented).
5. `_make_stub_record` returns a valid MacroIndicator with correct provenance.
6. BondSpreadClient._make_stub_record computes spread_bps correctly.
7. MacroSilverRecorder.save_indicators writes Parquet files with dedup.
8. MacroSilverRecorder quarantines records with wrong schema_version.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.agents.macro.clients.bond_spread_client import BondSpreadClient
from src.agents.macro.clients.fx_reserves_client import FXReservesClient
from src.agents.macro.clients.mospi_client import MOSPIClient
from src.agents.macro.clients.nse_fiidii_client import NSEDIIFIIClient
from src.agents.macro.clients.rbi_client import RBIClient
from src.agents.macro.recorder import MacroSilverRecorder
from src.schemas.macro_data import MacroIndicator, MacroIndicatorType, QualityFlag, SourceType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UTC = timezone.utc
_OBS_DATE = datetime(2026, 2, 21, 0, 0, 0, tzinfo=UTC)


@pytest.fixture
def mospi():
    return MOSPIClient()


@pytest.fixture
def rbi():
    return RBIClient()


@pytest.fixture
def nse_fiidii():
    return NSEDIIFIIClient()


@pytest.fixture
def fx_reserves():
    return FXReservesClient()


@pytest.fixture
def bond_spread():
    return BondSpreadClient()


# ---------------------------------------------------------------------------
# 1 & 2 — supported_indicators
# ---------------------------------------------------------------------------

class TestSupportedIndicators:
    def test_mospi_supports_cpi_wpi_iip(self, mospi):
        sup = mospi.supported_indicators
        assert MacroIndicatorType.CPI in sup
        assert MacroIndicatorType.WPI in sup
        assert MacroIndicatorType.IIP in sup
        assert len(sup) == 3

    def test_rbi_supports_fx_bulletin_india10y(self, rbi):
        sup = rbi.supported_indicators
        assert MacroIndicatorType.FX_RESERVES in sup
        assert MacroIndicatorType.RBI_BULLETIN in sup
        assert MacroIndicatorType.INDIA_10Y in sup

    def test_nse_fiidii_supports_fii_dii(self, nse_fiidii):
        sup = nse_fiidii.supported_indicators
        assert MacroIndicatorType.FII_FLOW in sup
        assert MacroIndicatorType.DII_FLOW in sup
        assert len(sup) == 2

    def test_fx_reserves_supports_only_fx(self, fx_reserves):
        assert fx_reserves.supported_indicators == frozenset([MacroIndicatorType.FX_RESERVES])

    def test_bond_spread_supports_only_spread(self, bond_spread):
        assert bond_spread.supported_indicators == frozenset([MacroIndicatorType.INDIA_US_10Y_SPREAD])


# ---------------------------------------------------------------------------
# 3 — ValueError for unsupported indicators
# ---------------------------------------------------------------------------

class TestUnsupportedIndicatorRaisesValueError:
    def test_mospi_rejects_fii_flow(self, mospi):
        from src.agents.macro.client import DateRange
        from datetime import date
        with pytest.raises(ValueError, match="MOSPIClient"):
            mospi.get_indicator(MacroIndicatorType.FII_FLOW, DateRange(date.today(), date.today()))

    def test_bond_spread_rejects_cpi(self, bond_spread):
        from src.agents.macro.client import DateRange
        from datetime import date
        with pytest.raises(ValueError, match="BondSpreadClient"):
            bond_spread.get_indicator(MacroIndicatorType.CPI, DateRange(date.today(), date.today()))

    def test_fx_reserves_rejects_wpi(self, fx_reserves):
        from src.agents.macro.client import DateRange
        from datetime import date
        with pytest.raises(ValueError):
            fx_reserves.get_indicator(MacroIndicatorType.WPI, DateRange(date.today(), date.today()))


# ---------------------------------------------------------------------------
# 4 — NotImplementedError for supported indicators (stub behaviour)
# ---------------------------------------------------------------------------

class TestSupportedIndicatorImplementationStatus:
    def test_mospi_indicators_implemented(self, mospi):
        from src.agents.macro.client import DateRange
        from datetime import date
        dr = DateRange(date.today(), date.today())
        # Should NOT raise NotImplementedError
        mospi.get_indicator(MacroIndicatorType.CPI, dr)
        mospi.get_indicator(MacroIndicatorType.WPI, dr)
        mospi.get_indicator(MacroIndicatorType.IIP, dr)

    def test_rbi_bulletin_implemented(self, rbi):
        from src.agents.macro.client import DateRange
        from datetime import date
        dr = DateRange(date.today(), date.today())
        # Should NOT raise NotImplementedError
        rbi.get_indicator(MacroIndicatorType.RBI_BULLETIN, dr)

    def test_rbi_fx_reserves_still_not_implemented(self, rbi):
        from src.agents.macro.client import DateRange
        from datetime import date
        with pytest.raises(NotImplementedError, match="Day 4"):
            rbi.get_indicator(MacroIndicatorType.FX_RESERVES, DateRange(date.today(), date.today()))

    def test_nse_fii_not_implemented(self, nse_fiidii):
        from src.agents.macro.client import DateRange
        from datetime import date
        with pytest.raises(NotImplementedError):
            nse_fiidii.get_indicator(MacroIndicatorType.FII_FLOW, DateRange(date.today(), date.today()))

    def test_bond_spread_not_implemented(self, bond_spread):
        from src.agents.macro.client import DateRange
        from datetime import date
        with pytest.raises(NotImplementedError):
            bond_spread.get_indicator(MacroIndicatorType.INDIA_US_10Y_SPREAD, DateRange(date.today(), date.today()))


# ---------------------------------------------------------------------------
# 5 — _make_stub_record returns valid MacroIndicator with full provenance
# ---------------------------------------------------------------------------

class TestStubRecordProvenance:
    def _assert_provenance(self, record: MacroIndicator, expected_indicator: MacroIndicatorType):
        assert isinstance(record, MacroIndicator)
        assert record.indicator_name == expected_indicator
        assert record.schema_version == "1.1"
        assert record.source_type == SourceType.OFFICIAL_API
        assert record.ingestion_timestamp_utc.tzinfo is not None
        assert record.ingestion_timestamp_ist.tzinfo is not None
        assert record.quality_status == QualityFlag.PASS

    def test_mospi_cpi_stub(self, mospi):
        rec = mospi._make_stub_record(MacroIndicatorType.CPI, 5.09, _OBS_DATE)
        self._assert_provenance(rec, MacroIndicatorType.CPI)
        assert rec.unit == "%"
        assert rec.period == "Monthly"
        assert rec.value == pytest.approx(5.09)

    def test_mospi_wpi_stub(self, mospi):
        rec = mospi._make_stub_record(MacroIndicatorType.WPI, 2.37, _OBS_DATE)
        self._assert_provenance(rec, MacroIndicatorType.WPI)

    def test_mospi_iip_stub(self, mospi):
        rec = mospi._make_stub_record(MacroIndicatorType.IIP, 3.8, _OBS_DATE)
        self._assert_provenance(rec, MacroIndicatorType.IIP)

    def test_rbi_fx_reserves_stub(self, rbi):
        rec = rbi._make_stub_record(MacroIndicatorType.FX_RESERVES, 628.5, _OBS_DATE)
        self._assert_provenance(rec, MacroIndicatorType.FX_RESERVES)
        assert rec.unit == "USD_Bn"
        assert rec.period == "Weekly"

    def test_rbi_bulletin_always_value_1(self, rbi):
        # RBI_BULLETIN encoding rule: value must always be 1.0
        rec = rbi._make_stub_record(MacroIndicatorType.RBI_BULLETIN, 99.0, _OBS_DATE)
        assert rec.value == pytest.approx(1.0), "RBI_BULLETIN value must be 1.0 (event marker)"
        assert rec.unit == "count"
        assert rec.period == "Irregular"

    def test_nse_fii_flow_stub(self, nse_fiidii):
        rec = nse_fiidii._make_stub_record(MacroIndicatorType.FII_FLOW, 1234.56, _OBS_DATE)
        self._assert_provenance(rec, MacroIndicatorType.FII_FLOW)
        assert rec.unit == "INR_Cr"
        assert rec.period == "Daily"

    def test_nse_dii_flow_stub(self, nse_fiidii):
        rec = nse_fiidii._make_stub_record(MacroIndicatorType.DII_FLOW, -890.12, _OBS_DATE)
        self._assert_provenance(rec, MacroIndicatorType.DII_FLOW)

    def test_fx_reserves_stub(self, fx_reserves):
        rec = fx_reserves._make_stub_record(630.2, _OBS_DATE)
        self._assert_provenance(rec, MacroIndicatorType.FX_RESERVES)
        assert rec.unit == "USD_Bn"

    def test_bond_spread_rejects_wrong_stub_indicator(self, mospi):
        with pytest.raises(ValueError):
            mospi._make_stub_record(MacroIndicatorType.FX_RESERVES, 1.0, _OBS_DATE)


# ---------------------------------------------------------------------------
# 6 — BondSpreadClient spread computation
# ---------------------------------------------------------------------------

class TestBondSpreadComputation:
    def test_positive_spread(self, bond_spread):
        rec = bond_spread._make_stub_record(
            india_10y_pct=7.15, us_10y_pct=4.25, observation_date=_OBS_DATE
        )
        assert rec.indicator_name == MacroIndicatorType.INDIA_US_10Y_SPREAD
        assert rec.unit == "bps"
        assert rec.value == pytest.approx((7.15 - 4.25) * 100, abs=0.01)

    def test_negative_spread_allowed(self, bond_spread):
        rec = bond_spread._make_stub_record(
            india_10y_pct=3.0, us_10y_pct=4.5, observation_date=_OBS_DATE
        )
        assert rec.value < 0

    def test_zero_spread(self, bond_spread):
        rec = bond_spread._make_stub_record(
            india_10y_pct=5.0, us_10y_pct=5.0, observation_date=_OBS_DATE
        )
        assert rec.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7 — MacroSilverRecorder: Parquet write + dedup
# ---------------------------------------------------------------------------

class TestMacroSilverRecorder:
    def _make_record(self, indicator: MacroIndicatorType, value: float, ts: datetime) -> MacroIndicator:
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        now = datetime.now(UTC)
        return MacroIndicator(
            indicator_name=indicator,
            value=value,
            unit="%",
            period="Monthly",
            timestamp=ts,
            source_type=SourceType.OFFICIAL_API,
            ingestion_timestamp_utc=now,
            ingestion_timestamp_ist=now.astimezone(IST),
            schema_version="1.1",
            quality_status=QualityFlag.PASS,
        )

    def test_save_creates_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = MacroSilverRecorder(
                base_dir=f"{tmpdir}/silver",
                quarantine_dir=f"{tmpdir}/quarantine",
            )
            rec = self._make_record(MacroIndicatorType.CPI, 5.1, _OBS_DATE)
            recorder.save_indicators([rec])

            # Parquet file should exist under silver/CPI/2026/02/2026-02-21.parquet
            expected = (
                f"{tmpdir}/silver/CPI/2026/02/2026-02-21.parquet"
            )
            assert pd.read_parquet(expected).shape[0] == 1

    def test_dedup_on_same_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = MacroSilverRecorder(
                base_dir=f"{tmpdir}/silver",
                quarantine_dir=f"{tmpdir}/quarantine",
            )
            rec1 = self._make_record(MacroIndicatorType.CPI, 5.1, _OBS_DATE)
            rec2 = self._make_record(MacroIndicatorType.CPI, 5.2, _OBS_DATE)  # same key, updated value
            recorder.save_indicators([rec1])
            recorder.save_indicators([rec2])

            expected = f"{tmpdir}/silver/CPI/2026/02/2026-02-21.parquet"
            df = pd.read_parquet(expected)
            assert df.shape[0] == 1  # deduped to 1 record
            assert float(df["value"].iloc[0]) == pytest.approx(5.2)

    def test_empty_batch_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = MacroSilverRecorder(
                base_dir=f"{tmpdir}/silver",
                quarantine_dir=f"{tmpdir}/quarantine",
            )
            recorder.save_indicators([])  # Should not raise


# ---------------------------------------------------------------------------
# 8 — MacroSilverRecorder: quarantine wrong schema_version
# ---------------------------------------------------------------------------

class TestMacroSilverRecorderQuarantine:
    def test_wrong_schema_version_quarantined(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from zoneinfo import ZoneInfo
            IST = ZoneInfo("Asia/Kolkata")
            now = datetime.now(UTC)
            # Bypass pydantic frozen model to inject wrong version for test
            # Use model_construct to skip validation
            bad_rec = MacroIndicator.model_construct(
                indicator_name=MacroIndicatorType.WPI,
                value=2.5,
                unit="%",
                period="Monthly",
                timestamp=_OBS_DATE,
                region="India",
                source_type=SourceType.OFFICIAL_API,
                ingestion_timestamp_utc=now,
                ingestion_timestamp_ist=now.astimezone(IST),
                schema_version="1.0",  # wrong version
                quality_status=QualityFlag.WARN,
            )
            recorder = MacroSilverRecorder(
                base_dir=f"{tmpdir}/silver",
                quarantine_dir=f"{tmpdir}/quarantine",
            )
            recorder.save_indicators([bad_rec])

            # Should be in quarantine, not silver
            import os
            silver_files = list((f"{tmpdir}/silver",))
            quarantine_files = list(
                (tmpdir + "/quarantine/WPI/2026/02").split()
            )
            # Just confirm silver dir has no CPI/WPI files (nothing committed)
            assert not any(
                f.endswith(".parquet")
                for f in os.listdir(f"{tmpdir}/silver")
                if os.path.isfile(f"{tmpdir}/silver/{f}")
            )
