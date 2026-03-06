import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Sequence

from src.schemas.text_data import SourceType as TextSourceType
from src.schemas.text_sidecar import SourceRouteDetail
from src.agents.textual.services.pdf_service import PDFExtractor

logger = logging.getLogger(__name__)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return token or "document"


def _expand_pdf_paths(pdf_paths: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for candidate in pdf_paths:
        raw_path = Path(candidate).expanduser()
        path = raw_path if raw_path.is_absolute() else (Path.cwd() / raw_path).resolve()

        if path.is_dir():
            resolved.extend(sorted(path.glob("*.pdf")))
            continue

        if path.suffix.lower() == ".pdf":
            resolved.append(path)
            continue

        logger.warning("Ignoring non-PDF path in textual runtime config: %s", path)
    return resolved


def _read_pdf_documents(pdf_paths: Sequence[str]) -> list[tuple[Path, bytes]]:
    docs: list[tuple[Path, bytes]] = []
    for path in _expand_pdf_paths(pdf_paths):
        if not path.exists():
            logger.warning("Configured PDF path does not exist: %s", path)
            continue
        try:
            docs.append((path, path.read_bytes()))
        except OSError as exc:
            logger.warning("Failed to read PDF path %s: %s", path, exc)
    return docs


def _infer_transcript_symbol(path: Path) -> str:
    for token in re.split(r"[^A-Za-z0-9]+", path.stem):
        if not token:
            continue
        lower = token.lower()
        if re.fullmatch(r"q[1-4]", lower):
            continue
        if re.fullmatch(r"20\d{2}", token):
            continue
        if token.isalpha():
            return token.upper()
    return "UNKNOWN"


def _infer_transcript_quarter(path: Path) -> str:
    stem_lower = path.stem.lower()
    for quarter in ("q1", "q2", "q3", "q4"):
        if quarter in stem_lower:
            return quarter.upper()
    return "Q4"


def _infer_year(path: Path, fallback_year: int) -> int:
    match = re.search(r"(20\d{2})", path.stem)
    if not match:
        return fallback_year
    return int(match.group(1))


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

    def __init__(self, pdf_paths: Sequence[str] | None = None):
        self.pdf_extractor = PDFExtractor()
        self.pdf_paths = list(pdf_paths or [])

    @staticmethod
    def _path_url(path: Path) -> str:
        try:
            return path.resolve().as_uri()
        except ValueError:
            return str(path.resolve())

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now = as_of_utc or datetime.now(UTC)

        docs = _read_pdf_documents(self.pdf_paths)
        if docs:
            records: list[RawTextRecord] = []
            for pdf_path, pdf_bytes in docs:
                extraction = self.pdf_extractor.extract(pdf_bytes)
                source_token = _safe_token(pdf_path.stem)
                records.append(
                    RawTextRecord(
                        record_type="news_article",
                        source_name=self.source_name,
                        source_id=f"rbi_report_{source_token}",
                        timestamp=now,
                        content=extraction.text,
                        payload={
                            "headline": f"RBI Report Extract ({pdf_path.name})",
                            "publisher": "Reserve Bank of India",
                            "url": self._path_url(pdf_path),
                            "is_published": True,
                            "license_ok": True,
                            "extraction_quality_score": extraction.quality_score,
                            "pdf_quality_status": extraction.quality_status,
                            "pdf_extracted_char_count": extraction.metrics["extracted_char_count"],
                        },
                        source_type=self.source_type,
                        source_route_detail=self.source_route_detail,
                    )
                )
            return records

        mock_pdf_content = b"%PDF-1.4 ... RBI Bulletin Content ... Monetary Policy ..."
        extraction = self.pdf_extractor.extract(mock_pdf_content)

        return [
            RawTextRecord(
                record_type="news_article",
                source_name=self.source_name,
                source_id="rbi_report_feb_2026",
                timestamp=now,
                content=extraction.text,
                payload={
                    "headline": "RBI MPC Policy Update (Extracted)",
                    "publisher": "Reserve Bank of India",
                    "url": "https://rbi.org.in/reports/feb_2026.pdf",
                    "is_published": True,
                    "license_ok": True,
                    "extraction_quality_score": extraction.quality_score,
                    "pdf_quality_status": extraction.quality_status,
                    "pdf_extracted_char_count": extraction.metrics["extracted_char_count"],
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            )
        ]


class EarningsTranscriptAdapter(BaseTextAdapter):
    source_name = "earnings_transcripts"
    source_type = TextSourceType.OFFICIAL_API
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED

    def __init__(self, pdf_paths: Sequence[str] | None = None):
        self.pdf_extractor = PDFExtractor()
        self.pdf_paths = list(pdf_paths or [])

    @staticmethod
    def _path_url(path: Path) -> str:
        try:
            return path.resolve().as_uri()
        except ValueError:
            return str(path.resolve())

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now = as_of_utc or datetime.now(UTC)

        docs = _read_pdf_documents(self.pdf_paths)
        if docs:
            records: list[RawTextRecord] = []
            for pdf_path, pdf_bytes in docs:
                extraction = self.pdf_extractor.extract(pdf_bytes)
                source_token = _safe_token(pdf_path.stem)
                inferred_symbol = _infer_transcript_symbol(pdf_path)
                inferred_quarter = _infer_transcript_quarter(pdf_path)
                inferred_year = _infer_year(pdf_path, now.year)
                records.append(
                    RawTextRecord(
                        record_type="earnings_transcript",
                        source_name=self.source_name,
                        source_id=f"earnings_transcript_{source_token}",
                        timestamp=now,
                        content=extraction.text,
                        payload={
                            "symbol": inferred_symbol,
                            "quarter": inferred_quarter,
                            "year": inferred_year,
                            "url": self._path_url(pdf_path),
                            "is_published": True,
                            "license_ok": True,
                            "extraction_quality_score": extraction.quality_score,
                            "pdf_quality_status": extraction.quality_status,
                            "pdf_extracted_char_count": extraction.metrics["extracted_char_count"],
                        },
                        source_type=self.source_type,
                        source_route_detail=self.source_route_detail,
                    )
                )
            return records

        mock_pdf_content = b"%PDF-1.4 ... INFY Q3 Transcript ... Earnings Call ..."
        extraction = self.pdf_extractor.extract(mock_pdf_content)

        return [
            RawTextRecord(
                record_type="earnings_transcript",
                source_name=self.source_name,
                source_id="infosys_q3_2026",
                timestamp=now,
                content=extraction.text,
                payload={
                    "symbol": "INFY",
                    "quarter": "Q3",
                    "year": 2026,
                    "url": "https://infosys.com/investors/q3_2026.pdf",
                    "is_published": True,
                    "license_ok": True,
                    "extraction_quality_score": extraction.quality_score,
                    "pdf_quality_status": extraction.quality_status,
                    "pdf_extracted_char_count": extraction.metrics["extracted_char_count"],
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
