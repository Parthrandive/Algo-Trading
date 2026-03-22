"""
Day 13 End-to-End Validation Script
Runs source -> Bronze -> Silver with failover, then writes observability traces/metrics.
"""

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

# Ensure src in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sentinel.nsepython_client import NSEPythonClient
from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.sentinel.failover_client import FailoverSentinelClient
from src.agents.sentinel.config import load_default_sentinel_config
from src.agents.sentinel.recorder import SilverRecorder
from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.pipeline import SentinelIngestPipeline
from src.agents.sentinel.client import NSEClientInterface
from config.symbols import SENTINEL_CORE_SYMBOLS
from src.schemas.market_data import (
    Bar,
    CorporateAction,
    CorporateActionType,
    QualityFlag,
    SourceType,
    Tick,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("e2e_test")


class DeterministicFallbackClient(NSEClientInterface):
    """
    Local deterministic source used to keep integration tests reproducible.
    """

    def __init__(self):
        self.source_type = SourceType.FALLBACK_SCRAPER

    def get_stock_quote(self, symbol: str) -> Tick:
        now_utc = datetime.now(UTC)
        seed = sum(ord(char) for char in symbol) % 500
        price = 1000.0 + float(seed)
        return Tick(
            symbol=symbol,
            timestamp=now_utc,
            source_type=self.source_type,
            price=price,
            volume=10000 + seed,
            quality_status=QualityFlag.PASS,
        )

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
    ) -> list[Bar]:
        start_utc = start_date.astimezone(UTC)
        end_utc = end_date.astimezone(UTC)
        if end_utc <= start_utc:
            return []

        rows: list[Bar] = []
        cursor = start_utc.replace(minute=0, second=0, microsecond=0)
        index = 0
        while cursor <= end_utc:
            base = 1000.0 + (index % 40)
            rows.append(
                Bar(
                    symbol=symbol,
                    timestamp=cursor,
                    source_type=self.source_type,
                    interval=interval,
                    open=base,
                    high=base + 3.5,
                    low=base - 2.5,
                    close=base + 1.25,
                    volume=5000 + index * 10,
                    quality_status=QualityFlag.PASS,
                )
            )
            cursor += timedelta(hours=1)
            index += 1
        return rows

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[CorporateAction]:
        start_utc = start_date.astimezone(UTC)
        end_utc = end_date.astimezone(UTC)
        if end_utc <= start_utc:
            return []

        span = end_utc - start_utc
        offsets = [0.20, 0.40, 0.60, 0.80]
        ex_dates = [start_utc + timedelta(seconds=int(span.total_seconds() * offset)) for offset in offsets]

        return [
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.DIVIDEND,
                value=18.5,
                ex_date=ex_dates[0],
                record_date=ex_dates[0] + timedelta(days=1),
                quality_status=QualityFlag.PASS,
            ),
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.SPLIT,
                ratio="2:1",
                ex_date=ex_dates[1],
                record_date=ex_dates[1] + timedelta(days=1),
                quality_status=QualityFlag.PASS,
            ),
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.BONUS,
                ratio="1:1",
                ex_date=ex_dates[2],
                record_date=ex_dates[2] + timedelta(days=1),
                quality_status=QualityFlag.PASS,
            ),
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.RIGHTS,
                ratio="3:10",
                ex_date=ex_dates[3],
                record_date=ex_dates[3] + timedelta(days=1),
                quality_status=QualityFlag.PASS,
            ),
        ]


def _is_parser_failure(exc: Exception) -> bool:
    return isinstance(exc, (ValueError, TypeError, KeyError))


def main():
    logger.info("Starting Day 13 E2E Validation...")

    # 0. Load runtime config
    config = load_default_sentinel_config()

    # 1. Initialize Clients
    primary = NSEPythonClient()
    fallback = YFinanceClient()
    deterministic_fallback = DeterministicFallbackClient()

    # 2. Setup Failover Client
    client = FailoverSentinelClient(
        primary_client=primary,
        fallback_clients=[fallback, deterministic_fallback],
        failure_threshold=config.failover.failure_threshold,
        cooldown_seconds=config.failover.cooldown_seconds,
        recovery_success_threshold=config.failover.recovery_success_threshold,
        fallback_source_type=SourceType.FALLBACK_SCRAPER,
    )

    # 3. Setup Recorders + observability output directories
    data_dir = PROJECT_ROOT / "data" / "e2e_test"
    bronze_dir = data_dir / "bronze"
    silver_dir = data_dir / "silver"
    quarantine_dir = data_dir / "quarantine"
    metrics_dir = data_dir / "metrics"
    traces_dir = data_dir / "traces"
    metrics_file = metrics_dir / "ingest_metrics.json"
    traces_file = traces_dir / "ingest_trace.jsonl"

    for path in (bronze_dir, silver_dir, quarantine_dir, metrics_dir, traces_dir):
        path.mkdir(parents=True, exist_ok=True)

    bronze = BronzeRecorder(base_dir=str(bronze_dir))
    silver = SilverRecorder(base_dir=str(silver_dir), quarantine_dir=str(quarantine_dir))

    # 5. Initialize Pipeline
    pipeline = SentinelIngestPipeline(
        client=client,
        silver_recorder=silver,
        bronze_recorder=bronze,
        session_rules=config.session_rules,
    )

    # 6. Test Data Ingestion + trace spans
    symbols = list(SENTINEL_CORE_SYMBOLS)
    end_date = datetime.now(ZoneInfo("UTC"))
    start_date = end_date - timedelta(days=7)

    span_rows: list[dict] = []
    parser_failures = 0
    total_records = 0
    fallback_records = 0

    def run_step(symbol: str, step: str, callback):
        nonlocal parser_failures, total_records, fallback_records
        started = datetime.now(UTC)
        perf_started = time.perf_counter()

        status = "success"
        error_message = None
        records_count = 0

        try:
            result = callback()
            if isinstance(result, list):
                records_count = len(result)
                for item in result:
                    if hasattr(item, "source_type"):
                        total_records += 1
                        if item.source_type == SourceType.FALLBACK_SCRAPER:
                            fallback_records += 1
            else:
                records_count = 1
                if hasattr(result, "source_type"):
                    total_records += 1
                    if result.source_type == SourceType.FALLBACK_SCRAPER:
                        fallback_records += 1
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            if _is_parser_failure(exc):
                parser_failures += 1
            result = None

        duration_ms = (time.perf_counter() - perf_started) * 1000.0
        finished = datetime.now(UTC)
        span_rows.append(
            {
                "trace_id": f"{symbol}:{step}:{int(started.timestamp() * 1000)}",
                "symbol": symbol,
                "step": step,
                "status": status,
                "records_count": records_count,
                "duration_ms": round(duration_ms, 3),
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "error_message": error_message,
            }
        )

        if status == "failed":
            logger.error("Step failed [%s:%s]: %s", symbol, step, error_message)
        else:
            logger.info("Step succeeded [%s:%s] records=%s duration_ms=%.2f", symbol, step, records_count, duration_ms)

    for symbol in symbols:
        run_step(
            symbol,
            "ingest_historical",
            lambda symbol=symbol: pipeline.ingest_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1h",
            ),
        )
        run_step(symbol, "ingest_quote", lambda symbol=symbol: pipeline.ingest_quote(symbol=symbol))
        run_step(
            symbol,
            "ingest_corporate_actions",
            lambda symbol=symbol: pipeline.ingest_corporate_actions(
                symbol=symbol,
                start_date=start_date - timedelta(days=365 * 3),
                end_date=end_date,
            ),
        )

    with traces_file.open("w", encoding="utf-8") as f:
        for span in span_rows:
            f.write(json.dumps(span))
            f.write("\n")

    durations = [span["duration_ms"] for span in span_rows]
    successful_ops = len([span for span in span_rows if span["status"] == "success"])
    failed_ops = len(span_rows) - successful_ops
    avg_duration_ms = (sum(durations) / len(durations)) if durations else 0.0
    max_duration_ms = max(durations) if durations else 0.0
    fallback_pct = (fallback_records / total_records * 100.0) if total_records else 0.0

    metrics_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "total_operations": len(span_rows),
        "successful_operations": successful_ops,
        "failed_operations": failed_ops,
        "parser_failures": parser_failures,
        "avg_ingest_latency_ms": round(avg_duration_ms, 3),
        "max_ingest_latency_ms": round(max_duration_ms, 3),
        "total_records_observed": total_records,
        "fallback_records_observed": fallback_records,
        "fallback_percentage": round(fallback_pct, 3),
        "trace_file": str(traces_file),
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    logger.info("Day 13 E2E test completed. Output generated in %s", data_dir)
    logger.info("Observability metrics saved to %s", metrics_file)
    logger.info("Observability traces saved to %s", traces_file)


if __name__ == "__main__":
    main()
