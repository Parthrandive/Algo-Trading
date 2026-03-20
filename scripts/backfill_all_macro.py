"""
Backfill ALL macro indicators from public FRED CSV endpoints into the database.

Uses `curl` for HTTP downloads (Python urllib/requests have SSL issues with
Anaconda's OpenSSL on some networks).

Usage:
    # Dry-run (preview only):
    python scripts/backfill_all_macro.py --dry-run

    # Backfill everything from 2000 to today:
    python scripts/backfill_all_macro.py

    # Backfill specific indicators:
    python scripts/backfill_all_macro.py --indicators CPI WPI IIP

    # Custom date range:
    python scripts/backfill_all_macro.py --start 2010-01-01 --end 2025-12-31

Data sources (all public, no API key required):
    CPI                 → FRED: INDCPIALLMINMEI     (Monthly)
    WPI                 → FRED: WPIATT01INM661N     (Monthly)
    IIP                 → FRED: INDPRINTO01IXOBM    (Monthly)
    FX_RESERVES         → FRED: TRESEGINM052N       (Monthly)
    INDIA_US_10Y_SPREAD → FRED: INDIRLTLT01STM (India 10Y) + DGS10 (US 10Y)
    REPO_RATE           → akshare (existing backfill)
    US_10Y              → akshare (existing backfill)
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.schemas.macro_data import (
    MacroIndicator,
    MacroIndicatorType,
    QualityFlag,
    SourceType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# ── FRED series definitions ──────────────────────────────────────────────────
FRED_SERIES: dict[str, dict] = {
    "CPI": {
        "series_id": "INDCPIALLMINMEI",
        "indicator": MacroIndicatorType.CPI,
        "unit": "Index",
        "period": "Monthly",
    },
    "WPI": {
        "series_id": "WPIATT01INM661N",
        "indicator": MacroIndicatorType.WPI,
        "unit": "Index",
        "period": "Monthly",
    },
    "IIP": {
        "series_id": "INDPRINTO01IXOBM",
        "indicator": MacroIndicatorType.IIP,
        "unit": "Index",
        "period": "Monthly",
    },
    "FX_RESERVES": {
        "series_id": "TRESEGINM052N",
        "indicator": MacroIndicatorType.FX_RESERVES,
        "unit": "USD_Millions",
        "period": "Monthly",
    },
    # Bond spread is composite; handled separately
}

# Bond spread legs
INDIA_10Y_SERIES = "INDIRLTLT01STM"
US_10Y_SERIES = "DGS10"

ALL_INDICATORS = ["CPI", "WPI", "IIP", "FX_RESERVES", "INDIA_US_10Y_SPREAD"]


def _parse_date_arg(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Bad date '{value}'. Use YYYY-MM-DD.") from exc


def _curl_fetch_csv(series_id: str) -> str:
    """Download a FRED CSV via curl (bypasses Python SSL issues)."""
    url = f"{FRED_CSV_BASE}?id={series_id}"
    result = subprocess.run(
        ["curl", "-sS", "--max-time", "60", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {series_id}: {result.stderr}")
    return result.stdout


def _parse_fred_csv(
    raw_csv: str,
    indicator: MacroIndicatorType,
    unit: str,
    period: str,
    start: date,
    end: date,
    series_id: str,
) -> list[MacroIndicator]:
    """Parse a FRED CSV into MacroIndicator records, filtered by date range."""
    reader = csv.DictReader(io.StringIO(raw_csv))
    if not reader.fieldnames or len(reader.fieldnames) < 2:
        raise RuntimeError(f"Unexpected CSV shape for {series_id}: {reader.fieldnames}")

    date_col = reader.fieldnames[0]
    value_col = reader.fieldnames[1]
    now_utc = datetime.now(UTC)
    records: list[MacroIndicator] = []

    for row in reader:
        raw_date = (row.get(date_col) or "").strip()
        raw_value = (row.get(value_col) or "").strip()

        if not raw_date or not raw_value or raw_value == ".":
            continue

        try:
            obs_date = date.fromisoformat(raw_date)
        except ValueError:
            continue

        if obs_date < start or obs_date > end:
            continue

        try:
            value = float(raw_value)
        except ValueError:
            continue

        records.append(
            MacroIndicator(
                indicator_name=indicator,
                value=value,
                unit=unit,
                period=period,
                timestamp=datetime(obs_date.year, obs_date.month, obs_date.day, tzinfo=UTC),
                source_type=SourceType.FALLBACK_SCRAPER,
                ingestion_timestamp_utc=now_utc,
                schema_version="1.1",
                quality_status=QualityFlag.PASS,
            )
        )

    return records


def _fetch_simple_indicator(
    name: str, start: date, end: date
) -> list[MacroIndicator]:
    """Fetch a simple (single-series) indicator from FRED."""
    spec = FRED_SERIES[name]
    raw_csv = _curl_fetch_csv(spec["series_id"])
    return _parse_fred_csv(
        raw_csv,
        indicator=spec["indicator"],
        unit=spec["unit"],
        period=spec["period"],
        start=start,
        end=end,
        series_id=spec["series_id"],
    )


def _fetch_bond_spread(start: date, end: date) -> list[MacroIndicator]:
    """
    Compute INDIA_US_10Y_SPREAD from two FRED legs.

    India 10Y (monthly) joined with US 10Y (daily, forward-filled).
    spread_bps = (india_10y_pct - us_10y_pct) * 100
    """
    india_csv = _curl_fetch_csv(INDIA_10Y_SERIES)
    us_csv = _curl_fetch_csv(US_10Y_SERIES)

    # Parse India 10Y
    india_rows = _parse_fred_csv(
        india_csv,
        indicator=MacroIndicatorType.INDIA_10Y,
        unit="Percent",
        period="Monthly",
        start=start,
        end=end,
        series_id=INDIA_10Y_SERIES,
    )

    # Parse US 10Y
    us_rows = _parse_fred_csv(
        us_csv,
        indicator=MacroIndicatorType.US_10Y,
        unit="Percent",
        period="Daily",
        start=start,
        end=end,
        series_id=US_10Y_SERIES,
    )

    if not india_rows or not us_rows:
        logger.warning("Bond spread: missing one or both legs (India=%d, US=%d)", len(india_rows), len(us_rows))
        return []

    # Build US 10Y lookup (date → value), use most recent US value for each India date
    us_by_date = {r.timestamp.date(): r.value for r in us_rows}
    us_dates_sorted = sorted(us_by_date.keys())

    now_utc = datetime.now(UTC)
    spread_records: list[MacroIndicator] = []
    us_idx = 0
    latest_us: float | None = None

    for india_rec in sorted(india_rows, key=lambda r: r.timestamp):
        obs_date = india_rec.timestamp.date()

        # Advance US pointer to find most recent US value on or before this date
        while us_idx < len(us_dates_sorted) and us_dates_sorted[us_idx] <= obs_date:
            latest_us = us_by_date[us_dates_sorted[us_idx]]
            us_idx += 1

        if latest_us is None:
            continue

        spread_bps = round((india_rec.value - latest_us) * 100, 4)

        spread_records.append(
            MacroIndicator(
                indicator_name=MacroIndicatorType.INDIA_US_10Y_SPREAD,
                value=spread_bps,
                unit="bps",
                period="Monthly",
                timestamp=india_rec.timestamp,
                source_type=SourceType.FALLBACK_SCRAPER,
                ingestion_timestamp_utc=now_utc,
                schema_version="1.1",
                quality_status=QualityFlag.PASS,
            )
        )

    return spread_records


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill macro indicators from public FRED CSV endpoints.",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        choices=ALL_INDICATORS,
        default=ALL_INDICATORS,
        help=f"Indicators to backfill. Default: all ({', '.join(ALL_INDICATORS)})",
    )
    parser.add_argument("--start", type=_parse_date_arg, default=date(2000, 1, 1), help="Start (YYYY-MM-DD)")
    parser.add_argument("--end", type=_parse_date_arg, default=datetime.now(UTC).date(), help="End (YYYY-MM-DD)")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy DB URL override")
    parser.add_argument("--dry-run", action="store_true", help="Fetch only; skip DB writes")
    args = parser.parse_args(argv)

    if args.start > args.end:
        logger.error("start must be <= end.")
        return 2

    total_fetched = 0
    total_saved = 0
    results: dict[str, dict] = {}

    for name in args.indicators:
        logger.info("=" * 60)
        logger.info("Backfilling %s (%s → %s)", name, args.start, args.end)

        try:
            if name == "INDIA_US_10Y_SPREAD":
                records = _fetch_bond_spread(args.start, args.end)
            else:
                records = _fetch_simple_indicator(name, args.start, args.end)
        except Exception as exc:
            logger.error("FAILED to fetch %s: %s", name, exc)
            results[name] = {"status": "FETCH_FAILED", "error": str(exc), "count": 0}
            continue

        if not records:
            logger.warning("No records for %s", name)
            results[name] = {"status": "NO_DATA", "count": 0}
            continue

        total_fetched += len(records)
        first_ts = min(r.timestamp for r in records)
        last_ts = max(r.timestamp for r in records)
        logger.info("Fetched %d %s records (%s → %s)", len(records), name, first_ts.date(), last_ts.date())

        if args.dry_run:
            logger.info("[DRY-RUN] Skipping DB write for %s", name)
            results[name] = {"status": "DRY_RUN", "count": len(records), "first": str(first_ts.date()), "last": str(last_ts.date())}
            continue

        try:
            from src.db.silver_db_recorder import SilverDBRecorder

            db_url = args.database_url or os.getenv(
                "DATABASE_URL",
                "postgresql://sentinel:sentinel@localhost:5432/sentinel_db",
            )
            recorder = SilverDBRecorder(database_url=db_url)
            recorder.save_macro_indicators(records)
            total_saved += len(records)
            logger.info("Saved %d %s records to database.", len(records), name)
            results[name] = {"status": "OK", "count": len(records), "first": str(first_ts.date()), "last": str(last_ts.date())}
        except Exception as exc:
            logger.error("FAILED to save %s: %s", name, exc)
            results[name] = {"status": "SAVE_FAILED", "error": str(exc), "count": len(records)}

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 60)
    for name, info in results.items():
        first_last = ""
        if "first" in info:
            first_last = f" | {info['first']} → {info['last']}"
        logger.info("  %-25s | %-12s | %4d records%s", name, info["status"], info["count"], first_last)
    logger.info("Total fetched: %d | Total saved: %d", total_fetched, total_saved)

    failed = [n for n, i in results.items() if i["status"] in ("FETCH_FAILED", "SAVE_FAILED")]
    if failed:
        logger.warning("FAILED: %s", failed)
        if args.dry_run:
            logger.warning("Dry-run mode: returning success despite fetch/save failures.")
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
