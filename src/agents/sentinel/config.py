from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

from src.schemas.market_data import SourceType

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "configs" / "nse_sentinel_runtime_v1.json"


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=0)
    base_backoff_seconds: float = Field(default=2.0, ge=0)


class RateLimitPolicy(BaseModel):
    calls: int = Field(default=1, ge=1)
    period_seconds: int = Field(default=1, ge=1)


class DataSourceConfig(BaseModel):
    name: str
    source_type: SourceType
    priority: int = Field(ge=1)  # 1 is highest
    enabled: bool = True
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: float = Field(default=10.0, gt=0)
    retries: RetryPolicy = Field(default_factory=RetryPolicy)
    rate_limit: RateLimitPolicy = Field(default_factory=RateLimitPolicy)


class SymbolUniverseConfig(BaseModel):
    version: str
    core_symbols: List[str]
    fx_symbols: List[str] = Field(default_factory=list)
    index_symbols: List[str] = Field(default_factory=list)
    review_cadence: str = "quarterly"

    @property
    def all_symbols(self) -> List[str]:
        symbols = [*self.core_symbols, *self.fx_symbols, *self.index_symbols]
        deduped: list[str] = []
        for symbol in symbols:
            if symbol not in deduped:
                deduped.append(symbol)
        return deduped


class SessionRules(BaseModel):
    timezone: str = "Asia/Kolkata"
    trading_days: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    regular_open: time = Field(default=time(hour=9, minute=15))
    regular_close: time = Field(default=time(hour=15, minute=30))
    pre_open_start: Optional[time] = Field(default=time(hour=9, minute=0))
    post_close_end: Optional[time] = None
    holidays: List[date] = Field(default_factory=list)

    @field_validator("trading_days")
    @classmethod
    def validate_trading_days(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("trading_days cannot be empty")
        if any(day < 0 or day > 6 for day in value):
            raise ValueError("trading_days must contain values between 0 (Mon) and 6 (Sun)")
        return value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def is_trading_session(self, dt: datetime) -> bool:
        if dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware for session checks")

        local_dt = dt.astimezone(self.tz)
        if local_dt.date() in self.holidays:
            return False
        if local_dt.weekday() not in self.trading_days:
            return False
        return self.regular_open <= local_dt.time() <= self.regular_close


class FailoverPolicy(BaseModel):
    failure_threshold: int = Field(default=2, ge=1)
    cooldown_seconds: int = Field(default=60, ge=0)
    recovery_success_threshold: int = Field(default=2, ge=1)


class HistoricalQualityConfig(BaseModel):
    min_rows_by_interval: dict[str, int] = Field(default_factory=lambda: {"1h": 1200, "1d": 300})
    min_history_days: int = Field(default=180, ge=1)
    min_coverage_pct: float = Field(default=60.0, ge=0.0, le=100.0)
    max_zero_volume_ratio: float = Field(default=0.35, ge=0.0, le=1.0)


class LiveQualityConfig(BaseModel):
    staleness_threshold_seconds: int = Field(default=300, ge=1)


class QualityGateConfig(BaseModel):
    historical: HistoricalQualityConfig = Field(default_factory=HistoricalQualityConfig)
    live: LiveQualityConfig = Field(default_factory=LiveQualityConfig)


class SentinelConfig(BaseModel):
    version: str = "nse-sentinel-runtime-v1"
    sources: List[DataSourceConfig]
    symbol_universe: SymbolUniverseConfig
    session_rules: SessionRules
    polling_interval_seconds: int = Field(default=60, ge=1)
    clock_drift_threshold_seconds: float = Field(default=0.5, ge=0)
    monotonicity_scope: str = Field(default="symbol_interval")
    failover: FailoverPolicy = Field(default_factory=FailoverPolicy)
    quality_gates: QualityGateConfig = Field(default_factory=QualityGateConfig)

    @model_validator(mode="after")
    def validate_source_priorities(self):
        enabled_priorities = [source.priority for source in self.sources if source.enabled]
        if len(enabled_priorities) != len(set(enabled_priorities)):
            raise ValueError("enabled sources must have unique priorities")
        return self

    def active_sources(self) -> List[DataSourceConfig]:
        return sorted([source for source in self.sources if source.enabled], key=lambda source: source.priority)


def load_sentinel_config(path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH) -> SentinelConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return SentinelConfig.model_validate(raw_config)


def load_default_sentinel_config() -> SentinelConfig:
    return load_sentinel_config(DEFAULT_RUNTIME_CONFIG_PATH)
