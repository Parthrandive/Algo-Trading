from datetime import UTC, datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

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
        if self.high < self.open:
            raise ValueError("High cannot be lower than Open")
        if self.high < self.low:
            raise ValueError("High cannot be lower than Low")
        if self.high < self.close:
            raise ValueError("High cannot be lower than Close")
        if self.low > self.open:
            raise ValueError("Low cannot be higher than Open")
        if self.low > self.close:
            raise ValueError("Low cannot be higher than Close")
        if self.low > self.high:
            raise ValueError("Low cannot be higher than High")
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
