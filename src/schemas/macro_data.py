from datetime import UTC, datetime
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

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

class MacroIndicatorType(str, Enum):
    CPI = "CPI"
    WPI = "WPI"
    IIP = "IIP"
    GDP = "GDP"
    INR_USD = "INR_USD"
    BRENT_CRUDE = "BRENT_CRUDE"
    US_10Y = "US_10Y"
    INDIA_10Y = "INDIA_10Y"
    FII_FLOW = "FII_FLOW"
    DII_FLOW = "DII_FLOW"
    REPO_RATE = "REPO_RATE"

class MacroIndicator(BaseModel):
    indicator_name: MacroIndicatorType
    value: float
    unit: str
    period: str # e.g., "Monthly", "Weekly", "Daily"
    timestamp: datetime
    region: str = Field(default="India")
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
