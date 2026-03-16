from datetime import UTC, datetime
from enum import Enum
import re
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from src.utils.time_sync import validate_utc_ist_consistency

IST = ZoneInfo("Asia/Kolkata")

class SourceType(str, Enum):
    OFFICIAL_API = "official_api"
    BROKER_API = "broker_api"
    FALLBACK_SCRAPER = "fallback_scraper"
    MANUAL_OVERRIDE = "manual_override"

class QualityFlag(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ObservationKind(str, Enum):
    QUOTE = "quote"
    BAR_UPDATE = "bar_update"
    FINAL_BAR = "final_bar"


class FreshnessStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


class MarketDataBase(BaseModel):
    symbol: str
    exchange: str = Field(default="NSE")
    timestamp: datetime
    source_type: SourceType = Field(validation_alias=AliasChoices("source_type", "source"))
    ingestion_timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        validation_alias=AliasChoices("ingestion_timestamp_utc", "ingestion_timestamp"),
    )
    ingestion_timestamp_ist: datetime = Field(default_factory=lambda: datetime.now(IST))
    schema_version: str = Field(default="1.0")
    quality_status: QualityFlag = Field(
        default=QualityFlag.PASS,
        validation_alias=AliasChoices("quality_status", "quality_flag"),
    )

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def populate_missing_ingestion_timestamps(cls, values):
        if not isinstance(values, dict):
            return values

        utc_value = values.get("ingestion_timestamp_utc", values.get("ingestion_timestamp"))
        ist_value = values.get("ingestion_timestamp_ist")

        if utc_value is None and ist_value is None:
            now_utc = datetime.now(UTC)
            values["ingestion_timestamp_utc"] = now_utc
            values["ingestion_timestamp_ist"] = now_utc.astimezone(IST)
            return values

        if utc_value is not None and ist_value is None and isinstance(utc_value, datetime) and utc_value.tzinfo is not None:
            values["ingestion_timestamp_ist"] = utc_value.astimezone(IST)

        if ist_value is not None and utc_value is None and isinstance(ist_value, datetime) and ist_value.tzinfo is not None:
            values["ingestion_timestamp_utc"] = ist_value.astimezone(UTC)

        return values

    @field_validator("ingestion_timestamp_utc")
    @classmethod
    def normalize_ingestion_timestamp_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("ingestion_timestamp_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("ingestion_timestamp_ist")
    @classmethod
    def normalize_ingestion_timestamp_ist(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("ingestion_timestamp_ist must be timezone-aware")
        return value.astimezone(IST)

    @model_validator(mode="after")
    def validate_utc_ist_pair(self):
        if not validate_utc_ist_consistency(
            self.ingestion_timestamp_utc,
            self.ingestion_timestamp_ist,
            tolerance_seconds=0.5,
        ):
            raise ValueError("ingestion_timestamp_utc and ingestion_timestamp_ist must represent the same instant")
        return self

    @property
    def source(self) -> SourceType:
        return self.source_type

    @property
    def ingestion_timestamp(self) -> datetime:
        return self.ingestion_timestamp_utc

    @property
    def quality_flag(self) -> QualityFlag:
        return self.quality_status

class Tick(MarketDataBase):
    price: float = Field(gt=0)
    volume: int = Field(ge=0)
    bid: Optional[float] = None
    ask: Optional[float] = None

class Bar(MarketDataBase):
    """
    Representation of a single OHLCV bar.
    Primary stored horizon: Hourly.
    Daily bars generated for confirmation.
    """
    interval: str  # e.g., "1m", "5m", "1h", "1d"
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    vwap: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_ohlc(self):
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")
        if self.close > self.high:
            raise ValueError("close must be less than or equal to high")
        if self.close < self.low:
            raise ValueError("close must be greater than or equal to low")
        return self

class CorporateActionType(str, Enum):
    DIVIDEND = "dividend"
    SPLIT = "split"
    BONUS = "bonus"
    RIGHTS = "rights"

class CorporateAction(MarketDataBase):
    action_type: CorporateActionType
    ratio: Optional[str] = None  # e.g., "1:2"
    value: Optional[float] = None # e.g., dividend amount
    ex_date: datetime
    record_date: Optional[datetime] = None

    @field_validator("ex_date")
    @classmethod
    def normalize_ex_date(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("ex_date must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("record_date")
    @classmethod
    def normalize_record_date(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("record_date must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("ratio")
    @classmethod
    def normalize_ratio(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().replace("/", ":")
        if not re.fullmatch(r"\d+(\.\d+)?:\d+(\.\d+)?", normalized):
            raise ValueError("ratio must follow '<num>:<num>' format, e.g. 1:2")
        return normalized

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("value must be greater than 0 when provided")
        return float(value)

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.record_date is not None and self.record_date < self.ex_date:
            raise ValueError("record_date cannot be earlier than ex_date")

        if self.action_type == CorporateActionType.DIVIDEND and self.value is None:
            raise ValueError("dividend actions require value")

        if self.action_type in {
            CorporateActionType.SPLIT,
            CorporateActionType.BONUS,
            CorporateActionType.RIGHTS,
        } and self.ratio is None:
            raise ValueError("split/bonus/rights actions require ratio")

        return self


class LiveMarketObservation(MarketDataBase):
    observation_kind: ObservationKind = Field(default=ObservationKind.QUOTE)
    interval: Optional[str] = None
    asset_type: str = Field(default="equity")
    source_name: Optional[str] = None
    source_status: str = Field(default="ok")
    freshness_status: FreshnessStatus = Field(default=FreshnessStatus.UNKNOWN)
    staleness_seconds: Optional[float] = Field(default=None, ge=0)
    last_price: Optional[float] = Field(default=None, gt=0)
    open: Optional[float] = Field(default=None, gt=0)
    high: Optional[float] = Field(default=None, gt=0)
    low: Optional[float] = Field(default=None, gt=0)
    close: Optional[float] = Field(default=None, gt=0)
    volume: Optional[int] = Field(default=None, ge=0)
    bid: Optional[float] = None
    ask: Optional[float] = None
    bar_timestamp: Optional[datetime] = None
    is_final_bar: bool = False

    @field_validator("bar_timestamp")
    @classmethod
    def normalize_bar_timestamp(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("bar_timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_live_observation(self):
        if self.last_price is None and self.close is None:
            raise ValueError("live observation requires last_price or close")

        if self.is_final_bar:
            if self.observation_kind != ObservationKind.FINAL_BAR:
                raise ValueError("finalized live bars must use observation_kind='final_bar'")
            if self.bar_timestamp is None or self.interval is None:
                raise ValueError("finalized live bars require bar_timestamp and interval")
            missing_ohlc = [
                name
                for name in ("open", "high", "low", "close")
                if getattr(self, name) is None
            ]
            if missing_ohlc:
                raise ValueError(f"finalized live bars require OHLC fields: {missing_ohlc}")
        return self
