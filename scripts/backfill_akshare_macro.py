import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.macro.client import DateRange
from src.agents.macro.clients.akshare_client import AkShareClient
from src.db.silver_db_recorder import SilverDBRecorder
from src.schemas.macro_data import MacroIndicatorType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_date(value: str):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill akshare macro indicators (REPO_RATE / US_10Y) into Silver DB."
    )
    parser.add_argument(
        "--indicator",
        choices=[MacroIndicatorType.REPO_RATE.value, MacroIndicatorType.US_10Y.value],
        default=MacroIndicatorType.REPO_RATE.value,
        help="Indicator to backfill. Default: REPO_RATE",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=datetime(2000, 1, 1, tzinfo=UTC).date(),
        help="Start date (YYYY-MM-DD). Default: 2000-01-01",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=datetime.now(UTC).date(),
        help="End date (YYYY-MM-DD). Default: today (UTC)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional SQLAlchemy database URL override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and normalize only; do not write to DB.",
    )
    args = parser.parse_args(argv)

    if args.start > args.end:
        logger.error("Invalid range: start date must be <= end date.")
        return 2

    date_range = DateRange(start=args.start, end=args.end)
    indicator = MacroIndicatorType(args.indicator)
    client = AkShareClient()

    try:
        records = list(client.get_indicator(indicator, date_range))
    except Exception as exc:
        logger.error("Failed to fetch %s from akshare: %s", indicator.value, exc)
        return 1

    if not records:
        logger.warning("No %s records found in range %s -> %s", indicator.value, args.start, args.end)
        return 0

    logger.info(
        "Prepared %d %s records (first=%s last=%s).",
        len(records),
        indicator.value,
        records[0].timestamp.isoformat(),
        records[-1].timestamp.isoformat(),
    )

    if args.dry_run:
        logger.info("Dry-run mode enabled. Skipping DB write.")
        return 0

    recorder = SilverDBRecorder(database_url=args.database_url)
    try:
        recorder.save_macro_indicators(records)
    except Exception as exc:
        logger.error("Failed to persist %s records: %s", indicator.value, exc)
        return 1

    logger.info("Successfully backfilled %d %s records into database.", len(records), indicator.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
