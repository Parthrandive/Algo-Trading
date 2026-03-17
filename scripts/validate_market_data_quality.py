import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.db.connection import get_engine


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate historical coverage and live freshness from market quality tables.")
    parser.add_argument("--symbols", type=str, help="Optional comma-separated symbol filter")
    parser.add_argument("--interval", default="1h", help="Interval to inspect")
    parser.add_argument("--report", type=str, help="Optional output report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    engine = get_engine()
    symbols = [value.strip() for value in args.symbols.split(",") if value.strip()] if args.symbols else None

    hist_query = """
        SELECT *
        FROM market_data_quality
        WHERE dataset_type = 'historical'
          AND interval = :interval
    """
    live_query = """
        SELECT *
        FROM market_data_quality
        WHERE dataset_type = 'live'
          AND interval = :interval
    """
    params = {"interval": args.interval}
    if symbols:
        hist_query += " AND symbol = ANY(:symbols)"
        live_query += " AND symbol = ANY(:symbols)"
        params["symbols"] = symbols
    hist_query += " ORDER BY symbol ASC"
    live_query += " ORDER BY symbol ASC"

    historical = pd.read_sql(text(hist_query), engine, params=params)
    live = pd.read_sql(text(live_query), engine, params=params)

    for frame in (historical, live):
        for column in ("first_timestamp", "last_timestamp", "updated_at"):
            if column in frame.columns and not frame.empty:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if "details_json" in frame.columns and not frame.empty:
            frame["details_json"] = frame["details_json"].apply(
                lambda value: None if value in (None, "") else json.loads(value)
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interval": args.interval,
        "historical": historical.to_dict(orient="records"),
        "live": live.to_dict(orient="records"),
        "summary": {
            "historical_train_ready": int((historical["train_ready"] == True).sum()) if "train_ready" in historical else 0,
            "historical_partial": int((historical["status"] == "partial").sum()) if "status" in historical else 0,
            "historical_failed": int((historical["status"] == "failed").sum()) if "status" in historical else 0,
            "live_fresh": int((live["status"] == "fresh").sum()) if "status" in live else 0,
            "live_stale": int((live["status"] == "stale").sum()) if "status" in live else 0,
            "live_failed": int((live["status"] == "failed").sum()) if "status" in live else 0,
        },
    }

    default_path = PROJECT_ROOT / "data" / "reports" / f"market_data_quality_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report_path = Path(args.report).expanduser().resolve() if args.report else default_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)

    print(f"Validation report written: {report_path}")
    print(
        "Historical summary: "
        f"train_ready={report['summary']['historical_train_ready']}, "
        f"partial={report['summary']['historical_partial']}, "
        f"failed={report['summary']['historical_failed']}"
    )
    print(
        "Live summary: "
        f"fresh={report['summary']['live_fresh']}, "
        f"stale={report['summary']['live_stale']}, "
        f"failed={report['summary']['live_failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
