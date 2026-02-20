import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.history import get_latest_local_timestamp, normalize_symbol

EXIT_SUCCESS = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_USAGE = 2
EXIT_FATAL = 3
EXIT_INTERRUPTED = 130

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "backfill" / "checkpoint.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "backfill" / "reports"

INTERVAL_SECONDS = {
    "1h": 3600,
    "1d": 86400,
}

TRANSIENT_ERROR_CODES = {"NETWORK_ERROR", "RATE_LIMITED", "UNKNOWN_ERROR"}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _parse_utc_date(date_str: str, end_of_day: bool) -> datetime:
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{date_str}'. Expected YYYY-MM-DD.") from exc

    if end_of_day:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    return parsed.replace(tzinfo=timezone.utc)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return ivalue


def _bounded_worker_count(value: str) -> int:
    ivalue = _positive_int(value)
    if ivalue > 16:
        raise argparse.ArgumentTypeError("workers must be <= 16")
    return ivalue


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} must be a JSON object.")
    return payload


def _classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if any(token in message for token in ["invalid", "not found", "unknown symbol", "delisted"]):
        return "INVALID_SYMBOL"
    if any(token in message for token in ["timed out", "timeout", "connection", "resolve", "temporary failure", "max retries"]):
        return "NETWORK_ERROR"
    if any(token in message for token in ["rate limit", "too many requests", "429"]):
        return "RATE_LIMITED"
    if any(token in message for token in ["permission", "forbidden", "unauthorized", "api key"]):
        return "CONFIG_ERROR"
    if "no data" in message:
        return "PROVIDER_NO_DATA"
    return "UNKNOWN_ERROR"


def _is_fresh(latest_ts: datetime | None, requested_end: datetime, skip_recent_hours: int) -> bool:
    if latest_ts is None:
        return False
    latest = _to_utc(latest_ts)
    freshness_cutoff = requested_end - timedelta(hours=skip_recent_hours)
    return latest >= freshness_cutoff


def _parse_checkpoint_timestamp(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None

    value = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _to_utc(parsed)


def _is_checkpoint_fresh(entry: dict, requested_end: datetime, skip_recent_hours: int) -> bool:
    if entry.get("status") != "SUCCESS":
        return False
    last_success_end = _parse_checkpoint_timestamp(entry.get("last_success_end_utc"))
    if last_success_end is None:
        return False
    return _is_fresh(last_success_end, requested_end, skip_recent_hours)


def _build_pipeline(write_bronze: bool):
    from src.agents.sentinel.broker_client import BrokerAPIClient
    from src.agents.sentinel.bronze_recorder import BronzeRecorder
    from src.agents.sentinel.config import load_default_sentinel_config
    from src.agents.sentinel.failover_client import FailoverSentinelClient
    from src.agents.sentinel.nsepython_client import NSEPythonClient
    from src.agents.sentinel.pipeline import SentinelIngestPipeline
    from src.agents.sentinel.recorder import SilverRecorder
    from src.agents.sentinel.yfinance_client import YFinanceClient

    primary = YFinanceClient()
    fallbacks = []

    broker_base_url = os.getenv("BROKER_API_BASE_URL")
    if broker_base_url:
        fallbacks.append(
            BrokerAPIClient(
                base_url=broker_base_url,
                api_key=os.getenv("BROKER_API_KEY"),
                access_token=os.getenv("BROKER_ACCESS_TOKEN"),
            )
        )

    fallbacks.append(NSEPythonClient())

    failover_client = FailoverSentinelClient(primary, fallbacks, failure_threshold=2, cooldown_seconds=60, recovery_success_threshold=2)
    config = load_default_sentinel_config()

    bronze_recorder = BronzeRecorder() if write_bronze else None
    return SentinelIngestPipeline(
        client=failover_client,
        silver_recorder=SilverRecorder(),
        bronze_recorder=bronze_recorder,
        session_rules=config.session_rules,
    )


def _result_record(
    *,
    symbol: str,
    status: str,
    requested_start: datetime,
    requested_end: datetime,
    interval: str,
    bars_fetched: int,
    attempt_count: int,
    message: str,
    error_code: str | None = None,
    error_message: str | None = None,
    last_success_end: datetime | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "status": status,
        "requested_start_utc": requested_start.isoformat(),
        "requested_end_utc": requested_end.isoformat(),
        "interval": interval,
        "bars_fetched": bars_fetched,
        "attempt_count": attempt_count,
        "message": message,
        "error_code": error_code,
        "error_message": error_message,
        "last_success_end_utc": last_success_end.isoformat() if last_success_end is not None else None,
        "last_attempt_utc": datetime.now(timezone.utc).isoformat(),
    }


def _checkpoint_entry(result: dict) -> dict:
    return {
        "symbol": result["symbol"],
        "status": result["status"],
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
        "attempt_count": result.get("attempt_count", 0),
        "last_attempt_utc": result.get("last_attempt_utc"),
        "last_success_end_utc": result.get("last_success_end_utc"),
        "bars_fetched": result.get("bars_fetched", 0),
        "requested_start_utc": result.get("requested_start_utc"),
        "requested_end_utc": result.get("requested_end_utc"),
        "interval": result.get("interval"),
    }


def _backfill_symbol(
    symbol: str,
    requested_start: datetime,
    requested_end: datetime,
    interval: str,
    *,
    skip_recent_hours: int,
    force_refresh: bool,
    write_bronze: bool,
    max_attempts: int,
    base_backoff_seconds: float,
) -> dict:
    symbol = normalize_symbol(symbol)
    latest_ts = get_latest_local_timestamp(symbol)

    if not force_refresh and _is_fresh(latest_ts, requested_end, skip_recent_hours):
        return _result_record(
            symbol=symbol,
            status="SKIPPED",
            requested_start=requested_start,
            requested_end=requested_end,
            interval=interval,
            bars_fetched=0,
            attempt_count=0,
            message="Local data is already fresh; skipping.",
            last_success_end=_to_utc(latest_ts) if latest_ts else None,
        )

    fetch_start = requested_start
    if not force_refresh and latest_ts is not None:
        latest_utc = _to_utc(latest_ts)
        if latest_utc >= fetch_start:
            fetch_start = latest_utc + timedelta(seconds=1)

    if fetch_start >= requested_end:
        return _result_record(
            symbol=symbol,
            status="SKIPPED",
            requested_start=requested_start,
            requested_end=requested_end,
            interval=interval,
            bars_fetched=0,
            attempt_count=0,
            message="No fetch window left after gap calculation; skipping.",
            last_success_end=_to_utc(latest_ts) if latest_ts else None,
        )

    pipeline = _build_pipeline(write_bronze=write_bronze)

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        # Small jitter per attempt across workers.
        time.sleep(random.uniform(0.1, 0.5))

        try:
            bars = pipeline.ingest_historical(symbol, fetch_start, requested_end, interval=interval)
            return _result_record(
                symbol=symbol,
                status="SUCCESS",
                requested_start=requested_start,
                requested_end=requested_end,
                interval=interval,
                bars_fetched=len(bars),
                attempt_count=attempt,
                message=f"Fetched {len(bars)} bars.",
                last_success_end=requested_end,
            )
        except Exception as exc:
            last_error = exc
            error_code = _classify_error(exc)
            if error_code in TRANSIENT_ERROR_CODES and attempt < max_attempts:
                backoff = (base_backoff_seconds * (2 ** (attempt - 1))) + random.uniform(0, 1)
                time.sleep(backoff)
                continue

            return _result_record(
                symbol=symbol,
                status="FAILED",
                requested_start=requested_start,
                requested_end=requested_end,
                interval=interval,
                bars_fetched=0,
                attempt_count=attempt,
                message="Fetch failed.",
                error_code=error_code,
                error_message=str(exc),
            )

    # Defensive fallback.
    return _result_record(
        symbol=symbol,
        status="FAILED",
        requested_start=requested_start,
        requested_end=requested_end,
        interval=interval,
        bars_fetched=0,
        attempt_count=max_attempts,
        message="Fetch failed.",
        error_code="UNKNOWN_ERROR",
        error_message=str(last_error) if last_error is not None else "Unknown failure",
    )


def _read_universe(universe_file: str) -> list[str]:
    path = Path(universe_file)
    if not path.exists():
        raise FileNotFoundError(f"Universe file '{universe_file}' not found.")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return []
        column_name = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), df.columns[0])
        values = df[column_name].dropna().astype(str).tolist()
    else:
        with path.open("r", encoding="utf-8") as handle:
            values = [line.strip() for line in handle if line.strip()]

    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bulk historical backfill for one or more symbols.")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g., TCS.NS,INFY.NS)")
    parser.add_argument("--universe", type=str, help="CSV/TXT file containing symbols")

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--days", type=_positive_int, default=30, help="Days to fetch when --from is not provided")
    window_group.add_argument("--from", dest="from_date", type=str, help="Start date (UTC) in YYYY-MM-DD")

    parser.add_argument("--to", dest="to_date", type=str, help="End date (UTC) in YYYY-MM-DD")
    parser.add_argument("--interval", choices=sorted(INTERVAL_SECONDS), default="1h", help="Historical interval")

    parser.add_argument("--workers", type=_bounded_worker_count, default=3, help="Worker threads (1-16)")
    parser.add_argument("--skip-recent-hours", type=_positive_int, default=1, help="Skip fetch if local data newer than this threshold")
    parser.add_argument("--max-attempts", type=_positive_int, default=3, help="Attempts per symbol for transient errors")

    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Checkpoint JSON file path")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Use checkpoint to skip fresh successful symbols")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore checkpoint state")
    parser.set_defaults(resume=True)

    parser.add_argument("--force-refresh", action="store_true", help="Ignore freshness checks and refetch full window")

    failure_group = parser.add_mutually_exclusive_group()
    failure_group.add_argument("--continue-on-error", dest="continue_on_error", action="store_true", help="Continue processing after failures")
    failure_group.add_argument("--fail-fast", dest="continue_on_error", action="store_false", help="Stop after first failure")
    parser.set_defaults(continue_on_error=True)

    parser.add_argument("--max-failures", type=_positive_int, help="Stop once this many failures occur")

    bronze_group = parser.add_mutually_exclusive_group()
    bronze_group.add_argument("--write-bronze", dest="write_bronze", action="store_true", help="Write Bronze events during backfill")
    bronze_group.add_argument("--no-write-bronze", dest="write_bronze", action="store_false", help="Skip Bronze writes during backfill")
    parser.set_defaults(write_bronze=False)

    parser.add_argument("--report", type=str, help="Report output path (default: data/backfill/reports/backfill_<timestamp>.json)")
    return parser


def _collect_symbols(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    symbols: list[str] = []

    if args.universe:
        symbols.extend(_read_universe(args.universe))

    if args.symbols:
        symbols.extend([value.strip() for value in args.symbols.split(",") if value.strip()])

    normalized = _dedupe_preserve_order([normalize_symbol(symbol) for symbol in symbols])
    if not normalized:
        parser.error("Provide --symbols and/or --universe with at least one symbol")

    return normalized


def _resolve_window(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[datetime, datetime]:
    requested_end = _parse_utc_date(args.to_date, end_of_day=True) if args.to_date else datetime.now(timezone.utc)

    if args.from_date:
        requested_start = _parse_utc_date(args.from_date, end_of_day=False)
    else:
        requested_start = requested_end - timedelta(days=args.days)

    if requested_start >= requested_end:
        parser.error("Requested start must be earlier than requested end")

    return requested_start, requested_end


def _print_symbol_result(result: dict) -> None:
    suffix = f" (bars={result['bars_fetched']}, attempts={result['attempt_count']})"
    line = f"[{result['symbol']}] {result['status']}: {result['message']}{suffix}"
    if result.get("error_code"):
        line += f" [error_code={result['error_code']}]"
    print(line)


def _run_backfill(
    *,
    symbols: list[str],
    requested_start: datetime,
    requested_end: datetime,
    interval: str,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> tuple[list[dict], bool]:
    checkpoint = _load_checkpoint(checkpoint_path) if args.resume else {}
    all_results: list[dict] = []

    pending_symbols: list[str] = []
    for symbol in symbols:
        state = checkpoint.get(symbol, {}) if args.resume else {}
        if not args.force_refresh and args.resume and _is_checkpoint_fresh(state, requested_end, args.skip_recent_hours):
            skipped = _result_record(
                symbol=symbol,
                status="SKIPPED",
                requested_start=requested_start,
                requested_end=requested_end,
                interval=interval,
                bars_fetched=0,
                attempt_count=0,
                message="Checkpoint indicates symbol is already fresh; skipping.",
                last_success_end=_parse_checkpoint_timestamp(state.get("last_success_end_utc")),
            )
            all_results.append(skipped)
            _print_symbol_result(skipped)
            continue
        pending_symbols.append(symbol)

    if not pending_symbols:
        return all_results, False

    failures = 0
    interrupted = False

    stop_after = args.max_failures if args.max_failures is not None else None
    sequential_mode = not args.continue_on_error or stop_after is not None

    if sequential_mode and args.workers != 1:
        print("Switching to single-worker mode to enforce stop conditions deterministically.")

    def process_result(result: dict) -> bool:
        nonlocal failures
        checkpoint[result["symbol"]] = _checkpoint_entry(result)
        _atomic_json_write(checkpoint_path, checkpoint)
        all_results.append(result)
        _print_symbol_result(result)

        if result["status"] == "FAILED":
            failures += 1
            if not args.continue_on_error:
                return True
            if stop_after is not None and failures >= stop_after:
                return True
        return False

    try:
        if sequential_mode:
            for symbol in pending_symbols:
                result = _backfill_symbol(
                    symbol,
                    requested_start,
                    requested_end,
                    interval,
                    skip_recent_hours=args.skip_recent_hours,
                    force_refresh=args.force_refresh,
                    write_bronze=args.write_bronze,
                    max_attempts=args.max_attempts,
                    base_backoff_seconds=2.0,
                )
                should_stop = process_result(result)
                if should_stop:
                    break
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(
                        _backfill_symbol,
                        symbol,
                        requested_start,
                        requested_end,
                        interval,
                        skip_recent_hours=args.skip_recent_hours,
                        force_refresh=args.force_refresh,
                        write_bronze=args.write_bronze,
                        max_attempts=args.max_attempts,
                        base_backoff_seconds=2.0,
                    ): symbol
                    for symbol in pending_symbols
                }

                for future in as_completed(future_map):
                    result = future.result()
                    process_result(result)
    except KeyboardInterrupt:
        interrupted = True

    return all_results, interrupted


def _build_report(
    *,
    args: argparse.Namespace,
    symbols: list[str],
    requested_start: datetime,
    requested_end: datetime,
    results: list[dict],
    interrupted: bool,
    started_at: datetime,
    finished_at: datetime,
) -> dict:
    summary = {
        "total_symbols": len(symbols),
        "processed": len(results),
        "success": sum(1 for row in results if row["status"] == "SUCCESS"),
        "failed": sum(1 for row in results if row["status"] == "FAILED"),
        "skipped": sum(1 for row in results if row["status"] == "SKIPPED"),
        "interrupted": interrupted,
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
    }

    return {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "requested_start_utc": requested_start.isoformat(),
        "requested_end_utc": requested_end.isoformat(),
        "interval": args.interval,
        "workers": args.workers,
        "resume": args.resume,
        "force_refresh": args.force_refresh,
        "write_bronze": args.write_bronze,
        "symbols": symbols,
        "summary": summary,
        "results": results,
    }


def run(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    symbols = _collect_symbols(args, parser)
    requested_start, requested_end = _resolve_window(args, parser)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    started_at = datetime.now(timezone.utc)

    print(
        f"Starting backfill: symbols={len(symbols)}, window={requested_start.isoformat()} -> {requested_end.isoformat()}, "
        f"interval={args.interval}, workers={args.workers}"
    )

    results, interrupted = _run_backfill(
        symbols=symbols,
        requested_start=requested_start,
        requested_end=requested_end,
        interval=args.interval,
        args=args,
        checkpoint_path=checkpoint_path,
    )

    finished_at = datetime.now(timezone.utc)

    default_report = DEFAULT_REPORT_DIR / f"backfill_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    report_path = Path(args.report).expanduser().resolve() if args.report else default_report

    report = _build_report(
        args=args,
        symbols=symbols,
        requested_start=requested_start,
        requested_end=requested_end,
        results=results,
        interrupted=interrupted,
        started_at=started_at,
        finished_at=finished_at,
    )
    _atomic_json_write(report_path, report)

    summary = report["summary"]
    print(
        "Backfill summary: "
        f"processed={summary['processed']}/{summary['total_symbols']}, "
        f"success={summary['success']}, failed={summary['failed']}, skipped={summary['skipped']}, "
        f"interrupted={summary['interrupted']}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Report: {report_path}")

    if interrupted:
        return EXIT_INTERRUPTED
    if summary["failed"] > 0:
        return EXIT_PARTIAL_FAILURE
    return EXIT_SUCCESS


def main() -> None:
    try:
        code = run(sys.argv[1:])
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Fatal error: {exc}")
        code = EXIT_FATAL

    raise SystemExit(code)


if __name__ == "__main__":
    main()
