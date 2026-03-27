from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.phase3_timescale import (
    apply_phase3_timescale_hypertables,
    format_phase3_timescale_result,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply Phase 3 Timescale hypertable migrations for strategic tables.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional SQLAlchemy database URL override.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first Timescale migration error instead of best-effort skip.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = apply_phase3_timescale_hypertables(
        database_url=args.database_url,
        strict=bool(args.strict),
    )
    print(format_phase3_timescale_result(result))
    return 0 if result["status"] in {"applied", "partial", "skipped", "skipped_non_postgres", "skipped_no_timescaledb"} else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main(sys.argv[1:]))
