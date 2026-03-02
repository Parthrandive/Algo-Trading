from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.schemas.text_data import SourceType as TextSourceType


class SourceRouteDetail(str, Enum):
    PRIMARY_API = "primary_api"
    OFFICIAL_FEED = "official_feed"
    FALLBACK_SCRAPER = "fallback_scraper"


class ComplianceStatus(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"


class TextSidecarMetadata(BaseModel):
    source_type: TextSourceType
    source_id: str
    ingestion_timestamp_utc: datetime
    source_route_detail: SourceRouteDetail
    quality_flags: list[str] = Field(default_factory=list)
    manipulation_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=0, ge=0)
    compliance_status: ComplianceStatus = ComplianceStatus.ALLOW
    compliance_reason: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("ingestion_timestamp_utc")
    @classmethod
    def normalize_ingestion_timestamp_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("ingestion_timestamp_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("quality_flags")
    @classmethod
    def normalize_quality_flags(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized_flags: list[str] = []
        for item in value:
            normalized = item.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_flags.append(normalized)
        return normalized_flags

    @model_validator(mode="after")
    def validate_compliance_reason(self) -> "TextSidecarMetadata":
        if self.compliance_status == ComplianceStatus.REJECT and not self.compliance_reason:
            raise ValueError("compliance_reason is required when compliance_status=reject")
        if self.compliance_status == ComplianceStatus.ALLOW and self.compliance_reason:
            raise ValueError("compliance_reason must be null when compliance_status=allow")
        return self

    @property
    def sidecar_key(self) -> tuple[TextSourceType, str, datetime]:
        return (self.source_type, self.source_id, self.ingestion_timestamp_utc)
