import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, Sequence

from src.schemas.text_data import SourceType as TextSourceType
from src.schemas.text_sidecar import SourceRouteDetail

logger = logging.getLogger(__name__)


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
        
        # Simulate a rate-limit retry scenario (mock logs)
        logger.info("XPostAdapter: Fetching posts using keyword templates from config...")
        
        return [
            RawTextRecord(
                record_type="social_post",
                source_name=self.source_name,
                source_id="x_post_india_market_1",
                timestamp=now,
                content="NIFTY 50 hits record high! FII inflows are surging today. #NSE #Nifty50",
                payload={
                    "platform": "X",
                    "likes": 1250,
                    "shares": 450,
                    "url": "https://x.com/market_news/status/india_1",
                    "is_published": True,
                    "license_ok": True,
                    "author": "market_pro_india",
                    "language": "en",
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            ),
            RawTextRecord(
                record_type="social_post",
                source_name=self.source_name,
                source_id="x_post_spam_2",
                timestamp=now,
                content="Get 1000% returns guaranteed! DM for jackpot tips. crypto giveaway #stocks",
                payload={
                    "platform": "X",
                    "likes": 10,
                    "shares": 2,
                    "url": "https://x.com/scammer/status/spam_2",
                    "is_published": True,
                    "license_ok": True,
                    "author": "jackpot_tips_xyz",
                    "language": "en",
                    "quality_flags": ["potential_spam"],
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            ),
            RawTextRecord(
                record_type="social_post",
                source_name=self.source_name,
                source_id="x_post_us_market_3",
                timestamp=now,
                content="S&P 500 continues to rally after Fed comments. US markets look strong.",
                payload={
                    "platform": "X",
                    "likes": 5000,
                    "shares": 1000,
                    "url": "https://x.com/us_markets/status/us_3",
                    "is_published": True,
                    "license_ok": True,
                    "author": "us_market_watcher",
                    "language": "en",
                    # This should be caught by India-relevance filter
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            )
        ]
