from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd
from sqlalchemy import text

from src.agents.sentinel.broker_client import BrokerAPIClient
from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.config import SentinelConfig, load_default_sentinel_config
from src.agents.sentinel.market_quality import HistoricalQualityThresholds, compute_symbol_quality
from src.agents.sentinel.market_utils import IST, infer_asset_type, normalize_timestamp
from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.technical.nsemine_fetcher import NseMineFetcher
from src.db.connection import get_engine
from src.db.models import Base
from src.db.silver_db_recorder import SilverDBRecorder
from src.schemas.market_data import Bar, QualityFlag, SourceType

_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False


def ensure_market_schema(database_url: str | None = None) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        engine = get_engine(database_url)
        Base.metadata.create_all(engine)
        _SCHEMA_READY = True


class HistoricalProvider(Protocol):
    name: str
    source_type: SourceType

    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> list[Bar]:
        ...


@dataclass
class ClientHistoricalProvider:
    name: str
    source_type: SourceType
    client: object

    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> list[Bar]:
        data = self.client.get_historical_data(symbol, start_date, end_date, interval=interval)
        return list(data or [])


@dataclass
class NSEMineHistoricalProvider:
    name: str = "nsemine"
    source_type: SourceType = SourceType.FALLBACK_SCRAPER

    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> list[Bar]:
        if infer_asset_type(symbol) != "equity":
            return []

        df = NseMineFetcher.fetch_historical(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval=interval,
        )
        if df.empty:
            return []

        bars: list[Bar] = []
        for row in df.to_dict(orient="records"):
            ts = pd.to_datetime(row["timestamp"], utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts.to_pydatetime(),
                    source_type=self.source_type,
                    interval=interval,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(float(row.get("volume", 0) or 0)),
                    quality_status=QualityFlag.PASS,
                )
            )
        return bars


def build_default_historical_providers(config: SentinelConfig | None = None) -> list[HistoricalProvider]:
    config = config or load_default_sentinel_config()
    providers: list[HistoricalProvider] = [NSEMineHistoricalProvider()]

    broker_base_url = os.getenv("BROKER_API_BASE_URL")
    if broker_base_url:
        providers.append(
            ClientHistoricalProvider(
                name="broker_rest",
                source_type=SourceType.BROKER_API,
                client=BrokerAPIClient(
                    base_url=broker_base_url,
                    api_key=os.getenv("BROKER_API_KEY"),
                    access_token=os.getenv("BROKER_ACCESS_TOKEN"),
                ),
            )
        )

    providers.append(
        ClientHistoricalProvider(
            name="yfinance",
            source_type=SourceType.OFFICIAL_API,
            client=YFinanceClient(),
        )
    )
    return providers


def _default_timezone(symbol: str):
    return IST if infer_asset_type(symbol) in {"equity", "index"} else datetime.now().astimezone().tzinfo or IST


def _bars_to_frame(symbol: str, interval: str, bars: list[Bar], *, source_name: str, source_rank: int) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "interval",
                "source_name",
                "source_type",
                "asset_type",
                "_source_rank",
            ]
        )

    rows: list[dict] = []
    default_tz = _default_timezone(symbol)
    asset_type = infer_asset_type(symbol)
    for bar in bars:
        ts = normalize_timestamp(bar.timestamp, default_timezone=default_tz)
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "interval": interval,
                "source_name": source_name,
                "source_type": bar.source_type.value,
                "asset_type": asset_type,
                "_source_rank": source_rank,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    frame["volume"] = frame["volume"].fillna(0).astype(int)
    frame = frame.sort_values(["timestamp", "_source_rank"]).reset_index(drop=True)
    return frame


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if existing.empty:
        duplicates = int(incoming.duplicated(subset=["timestamp"]).sum()) if not incoming.empty else 0
        merged = incoming.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        return merged, duplicates
    if incoming.empty:
        return existing, 0

    combined = pd.concat([existing, incoming], ignore_index=True)
    duplicates = int(combined.duplicated(subset=["timestamp"]).sum())
    merged = (
        combined.sort_values(["timestamp", "_source_rank"])
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return merged, duplicates


class HistoricalBackfillService:
    def __init__(
        self,
        *,
        providers: list[HistoricalProvider] | None = None,
        silver_recorder: SilverDBRecorder | None = None,
        bronze_recorder: BronzeRecorder | None = None,
        config: SentinelConfig | None = None,
        database_url: str | None = None,
    ):
        ensure_market_schema(database_url)
        self.config = config or load_default_sentinel_config()
        self.providers = providers or build_default_historical_providers(self.config)
        self.silver_recorder = silver_recorder or SilverDBRecorder(database_url)
        self.bronze_recorder = bronze_recorder
        self.engine = get_engine(database_url)
        self.thresholds = HistoricalQualityThresholds(
            min_rows_by_interval=dict(self.config.quality_gates.historical.min_rows_by_interval),
            min_history_days=self.config.quality_gates.historical.min_history_days,
            min_coverage_pct=self.config.quality_gates.historical.min_coverage_pct,
            max_zero_volume_ratio=self.config.quality_gates.historical.max_zero_volume_ratio,
        )

    def load_stored_bars(self, symbol: str, interval: str) -> pd.DataFrame:
        query = text(
            """
            SELECT symbol, timestamp, open, high, low, close, volume, interval, source_type
            FROM ohlcv_bars
            WHERE symbol = :symbol
              AND interval = :interval
            ORDER BY timestamp ASC
            """
        )
        frame = pd.read_sql(query, self.engine, params={"symbol": symbol, "interval": interval})
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame.dropna(subset=["timestamp"]).reset_index(drop=True)

    def build_quality_record(
        self,
        *,
        symbol: str,
        interval: str,
        frame: pd.DataFrame,
        requested_start: datetime,
        requested_end: datetime,
        source_names: list[str],
        source_type: str,
        duplicate_count: int = 0,
        dataset_type: str = "historical",
    ) -> dict:
        quality = compute_symbol_quality(
            frame=frame,
            symbol=symbol,
            interval=interval,
            session_rules=self.config.session_rules,
            thresholds=self.thresholds,
            duplicate_count=duplicate_count,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        return {
            "symbol": symbol,
            "interval": interval,
            "dataset_type": dataset_type,
            "exchange": "NSE",
            "asset_type": quality["asset_type"],
            "status": quality["status"],
            "train_ready": quality["train_ready"],
            "first_timestamp": quality["first_timestamp"],
            "last_timestamp": quality["last_timestamp"],
            "row_count": quality["row_count"],
            "duplicate_count": quality["duplicate_count"],
            "expected_rows": quality["expected_rows"],
            "missing_intervals": quality["missing_intervals"],
            "gap_count": quality["gap_count"],
            "largest_gap_intervals": quality["largest_gap_intervals"],
            "zero_volume_ratio": quality["zero_volume_ratio"],
            "coverage_pct": quality["coverage_pct"],
            "history_days": quality["history_days"],
            "source_name": ",".join(source_names) if source_names else None,
            "source_type": source_type,
            "details_json": {"quality_flags": quality["quality_flags"]},
            "updated_at": datetime.now(UTC),
        }

    def fetch_merged_history(
        self,
        *,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> tuple[pd.DataFrame, list[str], list[str], int]:
        merged = pd.DataFrame()
        source_names: list[str] = []
        errors: list[str] = []
        duplicate_count = 0

        for source_rank, provider in enumerate(self.providers):
            try:
                bars = provider.fetch_bars(symbol, start_date, end_date, interval)
            except Exception as exc:  # pragma: no cover - network/provider defensive path
                errors.append(f"{provider.name}:{exc}")
                continue

            frame = _bars_to_frame(symbol, interval, bars, source_name=provider.name, source_rank=source_rank)
            if frame.empty:
                continue

            merged, new_duplicates = _merge_frames(merged, frame)
            duplicate_count += new_duplicates
            source_names.append(provider.name)

            quality = compute_symbol_quality(
                frame=merged,
                symbol=symbol,
                interval=interval,
                session_rules=self.config.session_rules,
                thresholds=self.thresholds,
                duplicate_count=duplicate_count,
                requested_start=start_date,
                requested_end=end_date,
            )
            if quality["train_ready"]:
                break

        return merged, source_names, errors, duplicate_count

    def _persist_bronze(self, frame: pd.DataFrame) -> None:
        if self.bronze_recorder is None or frame.empty:
            return
        for row in frame.to_dict(orient="records"):
            event_time = pd.to_datetime(row["timestamp"], utc=True, errors="coerce")
            if pd.isna(event_time):
                continue
            self.bronze_recorder.save_event(
                source_id=row["source_name"],
                payload={key: value for key, value in row.items() if not key.startswith("_")},
                event_time=event_time.to_pydatetime(),
                symbol=row.get("symbol"),
                schema_id="market.bar.v2",
            )

    def _persist_bars(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        bars: list[Bar] = []
        for row in frame.to_dict(orient="records"):
            bars.append(
                Bar(
                    symbol=row["symbol"],
                    timestamp=pd.to_datetime(row["timestamp"], utc=True).to_pydatetime(),
                    source_type=SourceType(row["source_type"]),
                    interval=row["interval"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    quality_status=QualityFlag.PASS,
                )
            )
        self.silver_recorder.save_bars(bars)

    def backfill_symbol(
        self,
        *,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> dict:
        before_frame = self.load_stored_bars(symbol, interval)
        merged, source_names, errors, duplicate_count = self.fetch_merged_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        if not merged.empty:
            self._persist_bronze(merged)
            self._persist_bars(merged)

        after_frame = self.load_stored_bars(symbol, interval)
        quality_record = self.build_quality_record(
            symbol=symbol,
            interval=interval,
            frame=after_frame,
            requested_start=start_date,
            requested_end=end_date,
            source_names=source_names,
            source_type=merged["source_type"].iloc[0] if not merged.empty else None,
            duplicate_count=duplicate_count,
        )
        self.silver_recorder.save_market_data_quality([quality_record])

        after_status = quality_record["status"]
        if after_frame.empty:
            result_status = "FAILED"
        elif after_status == "train_ready":
            result_status = "SUCCESS"
        else:
            result_status = "PARTIAL"

        before_count = int(len(before_frame))
        after_count = int(len(after_frame))
        bars_added = max(0, after_count - before_count)
        message = (
            f"stored={after_count} rows, added={bars_added}, "
            f"coverage={quality_record['coverage_pct']}%, train_ready={quality_record['train_ready']}"
        )
        if errors:
            message += f"; source_errors={len(errors)}"

        return {
            "symbol": symbol,
            "status": result_status,
            "bars_fetched": int(len(merged)),
            "bars_added": bars_added,
            "before_row_count": before_count,
            "after_row_count": after_count,
            "sources_used": source_names,
            "source_errors": errors,
            "quality": quality_record,
            "message": message,
        }
