from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import nsepython
import pandas as pd
import yfinance as yf
from sqlalchemy import text

from src.agents.sentinel.broker_client import BrokerAPIClient
from src.agents.sentinel.config import SentinelConfig, load_default_sentinel_config
from src.agents.sentinel.historical_backfill import ensure_market_schema
from src.agents.sentinel.market_utils import IST, infer_asset_type, normalize_timestamp
from src.db.connection import get_engine
from src.db.silver_db_recorder import SilverDBRecorder
from src.schemas.market_data import (
    Bar,
    FreshnessStatus,
    LiveMarketObservation,
    ObservationKind,
    QualityFlag,
    SourceType,
    Tick,
)


class LiveProvider(Protocol):
    name: str
    source_type: SourceType

    def fetch_observation(self, symbol: str) -> LiveMarketObservation:
        ...


def _coerce_float(value):
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value):
    if value in (None, "", "-"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_intraday_high(raw: dict) -> float | None:
    intraday = raw.get("intraDayHighLow") or raw.get("dayHighLow") or {}
    return _coerce_float(intraday.get("max") or intraday.get("high"))


def _extract_intraday_low(raw: dict) -> float | None:
    intraday = raw.get("intraDayHighLow") or raw.get("dayHighLow") or {}
    return _coerce_float(intraday.get("min") or intraday.get("low"))


@dataclass
class NSEPythonLiveProvider:
    name: str = "nsepython"
    source_type: SourceType = SourceType.FALLBACK_SCRAPER

    def fetch_observation(self, symbol: str) -> LiveMarketObservation:
        if infer_asset_type(symbol) != "equity":
            raise ValueError(f"NSE live quote source only supports NSE equities: {symbol}")

        clean_symbol = symbol.replace(".NS", "").upper()
        payload = nsepython.nse_eq(clean_symbol)
        price_info = payload.get("priceInfo", {})

        last_price = _coerce_float(price_info.get("lastPrice") or price_info.get("close"))
        if last_price is None:
            raise ValueError(f"NSE quote payload missing price for {symbol}")

        volume = _coerce_int(
            payload.get("metadata", {}).get("totalTradedVolume")
            or payload.get("preOpenMarket", {}).get("totalTradedVolume")
            or payload.get("securityWiseDP", {}).get("quantityTraded")
        )

        return LiveMarketObservation(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            source_type=self.source_type,
            source_name=self.name,
            observation_kind=ObservationKind.QUOTE,
            asset_type=infer_asset_type(symbol),
            last_price=last_price,
            open=_coerce_float(price_info.get("open")),
            high=_extract_intraday_high(price_info),
            low=_extract_intraday_low(price_info),
            close=_coerce_float(price_info.get("close") or price_info.get("previousClose")),
            volume=volume,
            quality_status=QualityFlag.PASS,
        )


@dataclass
class YFinanceLiveProvider:
    name: str = "yfinance"
    source_type: SourceType = SourceType.OFFICIAL_API

    def fetch_observation(self, symbol: str) -> LiveMarketObservation:
        ticker = yf.Ticker(symbol)
        fast_info = ticker.fast_info
        price = getattr(fast_info, "last_price", None)
        if price is None:
            info = ticker.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            raise ValueError(f"YFinance quote missing price for {symbol}")

        return LiveMarketObservation(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            source_type=self.source_type,
            source_name=self.name,
            observation_kind=ObservationKind.QUOTE,
            asset_type=infer_asset_type(symbol),
            last_price=float(price),
            open=_coerce_float(getattr(fast_info, "open", None)),
            high=_coerce_float(getattr(fast_info, "day_high", None)),
            low=_coerce_float(getattr(fast_info, "day_low", None)),
            close=_coerce_float(getattr(fast_info, "previous_close", None)),
            volume=_coerce_int(getattr(fast_info, "last_volume", None)),
            quality_status=QualityFlag.PASS,
        )


@dataclass
class BrokerLiveProvider:
    client: BrokerAPIClient
    name: str = "broker_rest"
    source_type: SourceType = SourceType.BROKER_API

    def fetch_observation(self, symbol: str) -> LiveMarketObservation:
        tick = self.client.get_stock_quote(symbol)
        return LiveMarketObservation(
            symbol=symbol,
            timestamp=normalize_timestamp(tick.timestamp),
            source_type=self.source_type,
            source_name=self.name,
            observation_kind=ObservationKind.QUOTE,
            asset_type=infer_asset_type(symbol),
            last_price=tick.price,
            volume=tick.volume,
            bid=tick.bid,
            ask=tick.ask,
            quality_status=tick.quality_status,
        )


def build_default_live_providers(config: SentinelConfig | None = None) -> list[LiveProvider]:
    config = config or load_default_sentinel_config()
    providers: list[LiveProvider] = [NSEPythonLiveProvider()]

    broker_base_url = os.getenv("BROKER_API_BASE_URL")
    if broker_base_url:
        providers.append(
            BrokerLiveProvider(
                client=BrokerAPIClient(
                    base_url=broker_base_url,
                    api_key=os.getenv("BROKER_API_KEY"),
                    access_token=os.getenv("BROKER_ACCESS_TOKEN"),
                )
            )
        )

    providers.append(YFinanceLiveProvider())
    return providers


def finalize_live_observation_to_bar(
    observation: LiveMarketObservation,
    *,
    latest_finalized_bar_timestamp: datetime | None,
) -> Bar | None:
    if not observation.is_final_bar:
        return None
    if observation.observation_kind != ObservationKind.FINAL_BAR:
        return None
    if observation.bar_timestamp is None or observation.interval is None:
        return None

    bar_timestamp = normalize_timestamp(observation.bar_timestamp)
    if latest_finalized_bar_timestamp is not None:
        latest = normalize_timestamp(latest_finalized_bar_timestamp)
        if bar_timestamp <= latest:
            return None

    return Bar(
        symbol=observation.symbol,
        timestamp=bar_timestamp,
        source_type=observation.source_type,
        interval=observation.interval,
        open=float(observation.open),
        high=float(observation.high),
        low=float(observation.low),
        close=float(observation.close),
        volume=int(observation.volume or 0),
        quality_status=observation.quality_status,
    )


class LiveMarketIngestionService:
    def __init__(
        self,
        *,
        providers: list[LiveProvider] | None = None,
        silver_recorder: SilverDBRecorder | None = None,
        config: SentinelConfig | None = None,
        database_url: str | None = None,
    ):
        ensure_market_schema(database_url)
        self.config = config or load_default_sentinel_config()
        self.providers = providers or build_default_live_providers(self.config)
        self.silver_recorder = silver_recorder or SilverDBRecorder(database_url)
        self.engine = get_engine(database_url)
        self.staleness_threshold_seconds = self.config.quality_gates.live.staleness_threshold_seconds

    def _latest_finalized_bar_timestamp(self, symbol: str, interval: str) -> datetime | None:
        query = text(
            """
            SELECT MAX(timestamp) AS max_ts
            FROM ohlcv_bars
            WHERE symbol = :symbol
              AND interval = :interval
            """
        )
        with self.engine.connect() as conn:
            value = conn.execute(query, {"symbol": symbol, "interval": interval}).scalar()
        return None if value is None else normalize_timestamp(value)

    def _live_store_stats(self, symbol: str) -> tuple[int, datetime | None]:
        query = text(
            """
            SELECT COUNT(*) AS row_count, MAX(timestamp) AS max_ts
            FROM live_market_observations
            WHERE symbol = :symbol
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(query, {"symbol": symbol}).first()
        if row is None:
            return 0, None
        return int(row.row_count or 0), (None if row.max_ts is None else normalize_timestamp(row.max_ts))

    def _quality_payload(
        self,
        *,
        symbol: str,
        interval: str,
        observation: LiveMarketObservation | None,
        row_count: int,
        last_timestamp: datetime | None,
        source_error: str | None = None,
    ) -> dict:
        if observation is None or last_timestamp is None:
            status = "failed"
            freshness_status = FreshnessStatus.UNKNOWN.value
            staleness_seconds = None
            train_ready = False
            source_name = None
            source_type = None
            asset_type = infer_asset_type(symbol)
        else:
            staleness_seconds = max(0.0, (datetime.now(UTC) - normalize_timestamp(last_timestamp)).total_seconds())
            freshness_status = (
                FreshnessStatus.FRESH.value
                if staleness_seconds <= self.staleness_threshold_seconds
                else FreshnessStatus.STALE.value
            )
            status = "fresh" if freshness_status == FreshnessStatus.FRESH.value else "stale"
            train_ready = freshness_status == FreshnessStatus.FRESH.value
            source_name = observation.source_name
            source_type = observation.source_type.value
            asset_type = observation.asset_type

        return {
            "symbol": symbol,
            "interval": interval,
            "dataset_type": "live",
            "exchange": "NSE",
            "asset_type": asset_type,
            "status": status,
            "train_ready": train_ready,
            "first_timestamp": None,
            "last_timestamp": None if last_timestamp is None else last_timestamp.isoformat(),
            "row_count": row_count,
            "duplicate_count": 0,
            "expected_rows": None,
            "missing_intervals": None,
            "gap_count": None,
            "largest_gap_intervals": None,
            "zero_volume_ratio": None,
            "coverage_pct": None,
            "history_days": None,
            "source_name": source_name,
            "source_type": source_type,
            "details_json": {
                "source_status": None if observation is None else observation.source_status,
                "freshness_status": freshness_status,
                "staleness_seconds": staleness_seconds,
                "observation_kind": None if observation is None else observation.observation_kind.value,
                "is_final_bar": None if observation is None else observation.is_final_bar,
                "source_error": source_error,
            },
            "updated_at": datetime.now(UTC),
        }

    def poll_symbol(self, symbol: str, *, interval: str = "1h") -> dict:
        last_error: str | None = None
        observation: LiveMarketObservation | None = None
        used_provider_index: int | None = None

        for index, provider in enumerate(self.providers):
            try:
                observation = provider.fetch_observation(symbol)
                used_provider_index = index
                break
            except Exception as exc:  # pragma: no cover - provider/network path
                last_error = str(exc)
                continue

        if observation is None:
            row_count, last_ts = self._live_store_stats(symbol)
            quality = self._quality_payload(
                symbol=symbol,
                interval=interval,
                observation=None,
                row_count=row_count,
                last_timestamp=last_ts,
                source_error=last_error,
            )
            self.silver_recorder.save_market_data_quality([quality])
            return {
                "symbol": symbol,
                "status": "FAILED",
                "message": last_error or "All live providers failed",
                "quality": quality,
                "finalized_bar": False,
                "last_price": None,
                "volume": None,
                "source_name": None,
                "timestamp": None,
                "freshness_status": quality["details_json"]["freshness_status"],
                "source_status": None,
            }

        now_utc = datetime.now(UTC)
        staleness_seconds = max(0.0, (now_utc - normalize_timestamp(observation.timestamp)).total_seconds())
        freshness_status = (
            FreshnessStatus.FRESH
            if staleness_seconds <= self.staleness_threshold_seconds
            else FreshnessStatus.STALE
        )

        if used_provider_index is not None and used_provider_index > 0:
            observation = observation.model_copy(
                update={
                    "source_status": "fallback",
                    "quality_status": QualityFlag.WARN,
                }
            )
        observation = observation.model_copy(
            update={
                "freshness_status": freshness_status,
                "staleness_seconds": staleness_seconds,
            }
        )

        self.silver_recorder.save_live_observations([observation])
        if observation.last_price is not None:
            self.silver_recorder.save_ticks(
                [
                    Tick(
                        symbol=observation.symbol,
                        timestamp=observation.timestamp,
                        source_type=observation.source_type,
                        price=observation.last_price,
                        volume=int(observation.volume or 0),
                        bid=observation.bid,
                        ask=observation.ask,
                        quality_status=observation.quality_status,
                    )
                ]
            )

        finalized = False
        bar = finalize_live_observation_to_bar(
            observation,
            latest_finalized_bar_timestamp=self._latest_finalized_bar_timestamp(symbol, interval),
        )
        if bar is not None:
            self.silver_recorder.save_bars([bar])
            finalized = True

        row_count, last_ts = self._live_store_stats(symbol)
        quality = self._quality_payload(
            symbol=symbol,
            interval=interval,
            observation=observation,
            row_count=row_count,
            last_timestamp=last_ts,
        )
        self.silver_recorder.save_market_data_quality([quality])

        status = "SUCCESS" if quality["status"] == "fresh" else "PARTIAL"
        return {
            "symbol": symbol,
            "status": status,
            "message": f"source={observation.source_name}, freshness={quality['status']}",
            "quality": quality,
            "finalized_bar": finalized,
            "last_price": observation.last_price,
            "volume": observation.volume,
            "source_name": observation.source_name,
            "timestamp": observation.timestamp.isoformat(),
            "freshness_status": observation.freshness_status.value,
            "source_status": observation.source_status,
        }

    def poll_universe(
        self,
        symbols: list[str],
        *,
        interval: str = "1h",
        sleep_seconds: float | None = None,
        cycles: int = 1,
    ) -> list[dict]:
        results: list[dict] = []
        cycles = max(1, int(cycles))
        for cycle in range(cycles):
            for symbol in symbols:
                results.append(self.poll_symbol(symbol, interval=interval))
            if cycle < cycles - 1 and sleep_seconds:
                time.sleep(sleep_seconds)
        return results
