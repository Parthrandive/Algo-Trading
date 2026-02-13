from datetime import UTC, datetime
from enum import Enum
from typing import Optional, List
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

IST = ZoneInfo("Asia/Kolkata")

class SourceType(str, Enum):
    OFFICIAL_API = "official_api"
    BROKER_API = "broker_api"
    FALLBACK_SCRAPER = "fallback_scraper"
    MANUAL_OVERRIDE = "manual_override"
    RSS_FEED = "rss_feed"
    SOCIAL_MEDIA = "social_media"

class QualityFlag(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

class Language(str, Enum):
    EN = "en"
    HI = "hi"
    CM = "code_mixed" # Hinglish

class TextDataBase(BaseModel):
    source_id: str # Unique ID from source
    timestamp: datetime
    content: str
    url: Optional[str] = None
    author: Optional[str] = None
    language: Language = Field(default=Language.EN)
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
    
    # Enriched fields (populated later by agents)
    sentiment_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    embedding: Optional[List[float]] = None
    entities: Optional[List[str]] = None

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

class NewsArticle(TextDataBase):
    headline: str
    publisher: str

class SocialPost(TextDataBase):
    platform: str # e.g., "X", "Reddit"
    likes: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)

class EarningsTranscript(TextDataBase):
    symbol: str
    quarter: str
    year: int
