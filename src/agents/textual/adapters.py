from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, Sequence

from src.schemas.text_data import SourceType as TextSourceType
from src.schemas.text_sidecar import SourceRouteDetail


@dataclass(frozen=True)
class RawTextRecord:
    record_type: str
    source_name: str
    source_id: str
    timestamp: datetime
    content: str
    payload: dict[str, object]
    source_type: TextSourceType
    source_route_detail: SourceRouteDetail


class TextSourceAdapter(Protocol):
    source_name: str

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        ...


class BaseTextAdapter:
    source_name = "base_source"
    source_type = TextSourceType.RSS_FEED
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        _ = as_of_utc
        return []


class NSENewsAdapter(BaseTextAdapter):
    source_name = "nse_news"
    source_type = TextSourceType.RSS_FEED
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now = as_of_utc or datetime.now(UTC)
        return [
            RawTextRecord(
                record_type="news_article",
                source_name=self.source_name,
                source_id="nse_news_101",
                timestamp=now,
                content="NSE expands its derivatives segment with new index additions.",
                payload={
                    "headline": "NSE Index Expansion",
                    "publisher": "NSE India",
                    "url": "https://nseindia.com/news/101",
                    "is_published": True,
                    "license_ok": True,
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            )
        ]


class EconomicTimesAdapter(BaseTextAdapter):
    source_name = "economic_times"
    source_type = TextSourceType.RSS_FEED
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now = as_of_utc or datetime.now(UTC)
        return [
            RawTextRecord(
                record_type="news_article",
                source_name=self.source_name,
                source_id="et_news_202",
                timestamp=now,
                content="Indian markets hit record high as FII inflows surge.",
                payload={
                    "headline": "Markets at Record High",
                    "publisher": "Economic Times",
                    "url": "https://economictimes.indiatimes.com/news/202",
                    "is_published": True,
                    "license_ok": True,
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            ),
            RawTextRecord(
                record_type="news_article",
                source_name=self.source_name,
                source_id="et_news_blocked_303",
                timestamp=now,
                content="Unlicensed content example for compliance testing.",
                payload={
                    "headline": "Compliance Test Case",
                    "publisher": "Economic Times",
                    "url": "https://economictimes.indiatimes.com/news/303",
                    "is_published": True,
                    "license_ok": False,  # Should be rejected
                },
                source_type=self.source_type,
                source_route_detail=SourceRouteDetail.FALLBACK_SCRAPER,
            ),
        ]


class RBIReportsAdapter(BaseTextAdapter):
    source_name = "rbi_reports"
    source_type = TextSourceType.OFFICIAL_API
    source_route_detail = SourceRouteDetail.PRIMARY_API

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now = as_of_utc or datetime.now(UTC)
        return [
            RawTextRecord(
                record_type="news_article", # Mapped to NewsArticle for simplicity during Day 2
                source_name=self.source_name,
                source_id="rbi_report_feb_2026",
                timestamp=now,
                content="MPC maintains repo rate at 6.5% citing inflation concerns.",
                payload={
                    "headline": "RBI MPC Policy Update",
                    "publisher": "Reserve Bank of India",
                    "url": "https://rbi.org.in/reports/feb_2026",
                    "is_published": True,
                    "license_ok": True,
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            )
        ]


class EarningsTranscriptAdapter(BaseTextAdapter):
    source_name = "earnings_transcripts"
    source_type = TextSourceType.OFFICIAL_API
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now = as_of_utc or datetime.now(UTC)
        return [
            RawTextRecord(
                record_type="earnings_transcript",
                source_name=self.source_name,
                source_id="infosys_q3_2026",
                timestamp=now,
                content="Infosys reports 5% revenue growth in constant currency.",
                payload={
                    "symbol": "INFY",
                    "quarter": "Q3",
                    "year": 2026,
                    "url": "https://infosys.com/investors/q3_2026",
                    "is_published": True,
                    "license_ok": True,
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            )
        ]


class XPostAdapter(BaseTextAdapter):
    source_name = "x_posts"
    source_type = TextSourceType.SOCIAL_MEDIA
    source_route_detail = SourceRouteDetail.PRIMARY_API

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now = as_of_utc or datetime.now(UTC)
        return [
            RawTextRecord(
                record_type="social_post",
                source_name=self.source_name,
                source_id="x_post_404",
                timestamp=now,
                content="NIFTY is looking extremely bullish today! #IndianMarkets #Nifty50",
                payload={
                    "platform": "X",
                    "likes": 500,
                    "shares": 120,
                    "url": "https://x.com/user/status/404",
                    "is_published": True,
                    "license_ok": True,
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            )
        ]
