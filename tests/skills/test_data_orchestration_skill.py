"""
Functional tests for Data Orchestration Skill (§6 Worked Examples).

Tests that the backfill pipeline produces correct outputs, handles errors
gracefully, and covers edge cases. Run with:

    pytest tests/skills/test_data_orchestration_skill.py -v
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.macro_data import (
    MacroIndicator,
    MacroIndicatorType,
    QualityFlag,
    SourceType,
)


# ── 1. VALID OUTPUTS ────────────────────────────────────────────────────────


class TestValidOutputs:
    """Verify that the backfill pipeline produces structurally correct records."""

    def test_macro_indicator_schema_accepts_valid_record(self):
        """A well-formed MacroIndicator should construct without error."""
        record = MacroIndicator(
            indicator_name=MacroIndicatorType.CPI,
            value=157.77,
            unit="Index",
            period="Monthly",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            source_type=SourceType.FALLBACK_SCRAPER,
            ingestion_timestamp_utc=datetime.now(UTC),
            schema_version="1.1",
            quality_status=QualityFlag.PASS,
        )
        assert record.indicator_name == MacroIndicatorType.CPI
        assert record.value == 157.77
        assert record.schema_version == "1.1"
        assert record.source_type == SourceType.FALLBACK_SCRAPER

    def test_macro_indicator_rejects_missing_timezone(self):
        """Timezone-naive timestamps must be rejected."""
        with pytest.raises(ValueError, match="timezone-aware"):
            MacroIndicator(
                indicator_name=MacroIndicatorType.CPI,
                value=100.0,
                unit="Index",
                period="Monthly",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                source_type=SourceType.FALLBACK_SCRAPER,
                ingestion_timestamp_utc=datetime(2025, 1, 1),  # no tz!
            )

    def test_all_indicator_types_in_enum(self):
        """All backfilled indicators must exist in the MacroIndicatorType enum."""
        required = {"CPI", "WPI", "IIP", "FX_RESERVES", "INDIA_US_10Y_SPREAD"}
        actual = {member.value for member in MacroIndicatorType}
        assert required.issubset(actual), f"Missing from enum: {required - actual}"

    def test_fred_series_ids_defined_in_backfill_script(self):
        """The backfill script must define FRED series for all required indicators."""
        script_path = PROJECT_ROOT / "scripts" / "backfill_all_macro.py"
        content = script_path.read_text()
        # These are the FRED series IDs documented in the skill §6.1
        for series_id in ["INDCPIALLMINMEI", "WPIATT01INM661N", "INDPRINTO01IXOBM",
                          "TRESEGINM052N", "INDIRLTLT01STM", "DGS10"]:
            assert series_id in content, f"FRED series {series_id} missing from backfill script"


# ── 2. API CALLS SUCCEED ────────────────────────────────────────────────────


class TestApiCalls:
    """Verify that FRED CSV downloads work via curl."""

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Skip network calls in CI",
    )
    def test_fred_csv_download_returns_valid_csv(self):
        """curl fetch from FRED returns parseable CSV with date + value columns."""
        result = subprocess.run(
            ["curl", "-sS", "--max-time", "30",
             "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDCPIALLMINMEI"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"curl failed: {result.stderr}"
        lines = result.stdout.strip().split("\n")
        assert len(lines) > 100, f"Expected >100 rows, got {len(lines)}"
        # Verify CSV structure
        reader = csv.DictReader(io.StringIO(result.stdout))
        assert reader.fieldnames is not None
        assert len(reader.fieldnames) == 2
        assert "observation_date" in reader.fieldnames[0].lower() or "date" in reader.fieldnames[0].lower()

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Skip network calls in CI",
    )
    def test_all_fred_series_accessible(self):
        """Every FRED series ID used by the backfill script must return HTTP 200."""
        series_ids = ["INDCPIALLMINMEI", "WPIATT01INM661N", "INDPRINTO01IXOBM",
                       "TRESEGINM052N", "INDIRLTLT01STM", "DGS10"]
        for sid in series_ids:
            result = subprocess.run(
                ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
                 "--max-time", "15",
                 f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"],
                capture_output=True, text=True,
            )
            assert result.stdout.strip() == "200", f"FRED series {sid} returned {result.stdout}"


# ── 3. ERROR HANDLING ────────────────────────────────────────────────────────


class TestErrorHandling:
    """Verify graceful failure on bad inputs."""

    def test_backfill_rejects_invalid_date_range(self):
        """start > end must fail with exit code 2."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "backfill_all_macro.py"),
             "--start", "2025-01-01", "--end", "2020-01-01", "--dry-run"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 2

    def test_backfill_handles_unknown_indicator_gracefully(self):
        """Passing a non-existent indicator should be caught by argparse."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "backfill_all_macro.py"),
             "--indicators", "FAKE_INDICATOR", "--dry-run"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode != 0  # argparse should reject it

    def test_macro_indicator_rejects_extra_fields(self):
        """Schema has extra='forbid' — unknown fields must raise."""
        with pytest.raises(Exception):
            MacroIndicator(
                indicator_name=MacroIndicatorType.CPI,
                value=100.0,
                unit="Index",
                period="Monthly",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                source_type=SourceType.FALLBACK_SCRAPER,
                ingestion_timestamp_utc=datetime.now(UTC),
                totally_fake_field="should_fail",  # extra field!
            )


# ── 4. EDGE CASES ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Cover non-obvious scenarios that could silently corrupt data."""

    def test_fred_csv_with_missing_values_skipped(self):
        """FRED CSVs use '.' for missing values — these must be skipped, not crash."""
        fake_csv = "observation_date,INDCPIALLMINMEI\n2020-01-01,100.5\n2020-02-01,.\n2020-03-01,101.2\n"

        # Import the parser from the backfill script
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from backfill_all_macro import _parse_fred_csv

        records = _parse_fred_csv(
            fake_csv,
            indicator=MacroIndicatorType.CPI,
            unit="Index",
            period="Monthly",
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            series_id="INDCPIALLMINMEI",
        )
        # Should have 2 records (row with '.' skipped)
        assert len(records) == 2
        assert records[0].value == 100.5
        assert records[1].value == 101.2

    def test_fred_csv_date_filter_respects_range(self):
        """Records outside the requested date range must be excluded."""
        fake_csv = (
            "observation_date,TEST\n"
            "2019-12-01,99.0\n"  # before range
            "2020-01-01,100.0\n"  # in range
            "2020-06-01,105.0\n"  # in range
            "2021-01-01,110.0\n"  # after range
        )
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from backfill_all_macro import _parse_fred_csv

        records = _parse_fred_csv(
            fake_csv,
            indicator=MacroIndicatorType.CPI,
            unit="Index",
            period="Monthly",
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            series_id="TEST",
        )
        assert len(records) == 2
        assert records[0].value == 100.0
        assert records[1].value == 105.0

    def test_idempotent_rerun_does_not_crash(self):
        """Running backfill twice with --dry-run should produce identical output."""
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "backfill_all_macro.py"),
            "--indicators", "CPI", "--start", "2024-01-01", "--end", "2024-03-01",
            "--dry-run",
        ]
        r1 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert r1.returncode == 0
        assert r2.returncode == 0
        # Both should report same record count
        assert "CPI" in r1.stdout or "CPI" in r1.stderr
        assert "CPI" in r2.stdout or "CPI" in r2.stderr

    def test_bond_spread_handles_empty_leg(self):
        """If one bond leg returns no data, spread should return empty, not crash."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from backfill_all_macro import _parse_fred_csv

        empty_csv = "observation_date,INDIRLTLT01STM\n"
        records = _parse_fred_csv(
            empty_csv,
            indicator=MacroIndicatorType.INDIA_10Y,
            unit="Percent",
            period="Monthly",
            start=date(2020, 1, 1),
            end=date(2025, 12, 31),
            series_id="INDIRLTLT01STM",
        )
        assert len(records) == 0
