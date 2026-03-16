import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.sentinel.config import load_default_sentinel_config
from src.agents.sentinel.live_market import LiveMarketIngestionService
from src.utils.history import normalize_symbol


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    tmp_path.replace(path)


def _read_universe(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return []
        column = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), df.columns[0])
        return df[column].dropna().astype(str).tolist()
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _collect_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.universe:
        symbols.extend(_read_universe(Path(args.universe)))
    if args.symbols:
        symbols.extend([value.strip() for value in args.symbols.split(",") if value.strip()])
    if not symbols:
        symbols = load_default_sentinel_config().symbol_universe.all_symbols
    normalized: list[str] = []
    for symbol in symbols:
        value = normalize_symbol(symbol)
        if value not in normalized:
            normalized.append(value)
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poll live NSE/market data into the normalized live observation store.")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--universe", type=str, help="CSV/TXT file of symbols")
    parser.add_argument("--interval", default="1h", help="Target finalized bar interval when supported")
    parser.add_argument("--cycles", type=int, default=1, help="Number of polling cycles")
    parser.add_argument("--sleep-seconds", type=float, default=None, help="Sleep between cycles")
    parser.add_argument("--report", type=str, help="Optional output report JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    symbols = _collect_symbols(args)

    service = LiveMarketIngestionService()
    started_at = datetime.now(timezone.utc)
    results = service.poll_universe(
        symbols,
        interval=args.interval,
        sleep_seconds=args.sleep_seconds,
        cycles=max(1, int(args.cycles)),
    )
    finished_at = datetime.now(timezone.utc)

    report = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "interval": args.interval,
        "symbols": symbols,
        "summary": {
            "total_symbols": len(symbols),
            "success": sum(1 for row in results if row["status"] == "SUCCESS"),
            "partial": sum(1 for row in results if row["status"] == "PARTIAL"),
            "failed": sum(1 for row in results if row["status"] == "FAILED"),
        },
        "results": results,
    }

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    default_path = PROJECT_ROOT / "data" / "live" / "reports" / f"live_ingestion_{timestamp}.json"
    report_path = Path(args.report).expanduser().resolve() if args.report else default_path
    _atomic_json_write(report_path, report)

    print(
        "Live ingestion summary: "
        f"success={report['summary']['success']}, "
        f"partial={report['summary']['partial']}, "
        f"failed={report['summary']['failed']}"
    )
    print(f"Report: {report_path}")

    if report["summary"]["failed"] > 0 or report["summary"]["partial"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
