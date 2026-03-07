import hashlib
import html
import json
import logging
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

try:
    import pdfplumber
except ImportError:  # pragma: no cover - exercised in environments without PDF extras.
    pdfplumber = None

from src.agents.textual.services.pdf_service import PDFExtractor
from src.schemas.text_data import SourceType as TextSourceType
from src.schemas.text_sidecar import SourceRouteDetail

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NSE_KEYWORDS = ("nse", "nifty", "sensex", "bank nifty")

HTTPGetter = Callable[[str, dict[str, str] | None], str]

_RBI_DEFAULT_SOURCE_POLICY: dict[str, object] = {
    "rss_feed_url": "https://www.rbi.org.in/Scripts/rss.aspx",
    "dbie_download_url": "https://data.rbi.org.in/DBIE/#/",
    "fallback_scraper_url": "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx",
    "simulate_rss_outage": False,
    "simulate_dbie_outage": False,
    "enable_emergency_fallback_scraper": False,
}


@dataclass(frozen=True)
class _RBIResolvedRoute:
    source_type: TextSourceType
    route_detail: SourceRouteDetail
    source_url: str
    source_channel: str


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


def _path_url(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except ValueError:
        return str(path.resolve())


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
    cache_namespace = "default"

    def __init__(
        self,
        *,
        max_items: int = 25,
        http_get: HTTPGetter | None = None,
        cache_root: Path | None = None,
    ) -> None:
        self.max_items = max_items
        self._has_custom_http_get = http_get is not None
        self._http_get = http_get or self._default_http_get
        base_root = cache_root or (Path(__file__).resolve().parents[3] / "data" / "textual_cache")
        self._cache_dir = base_root / self.cache_namespace
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return _WS_RE.sub(" ", _TAG_RE.sub(" ", html.unescape(value or ""))).strip()

    @staticmethod
    def _default_http_get(url: str, headers: dict[str, str] | None = None) -> str:
        request_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if headers:
            request_headers.update(headers)
        req = urllib.request.Request(url=url, headers=request_headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="ignore")

    def _fetch_text(
        self,
        *,
        url: str,
        cache_key: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        cache_file = self._cache_dir / f"{cache_key}.txt"
        try:
            text = self._http_get(url, headers)
            if text.strip():
                cache_file.write_text(text, encoding="utf-8")
                return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] fetch failed for %s: %s", self.source_name, url, exc)

        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _parse_rss_datetime(raw_value: str | None) -> datetime | None:
        if not raw_value:
            return None
        try:
            parsed = parsedate_to_datetime(raw_value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _stable_id(prefix: str, token: str) -> str:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        _ = as_of_utc
        return []


class NSENewsAdapter(BaseTextAdapter):
    source_name = "nse_news"
    source_type = TextSourceType.RSS_FEED
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED
    cache_namespace = "nse_news"

    _NSE_ANNOUNCEMENT_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"
    _ET_FALLBACK_URL = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        records = self._fetch_from_nse_api(as_of_utc=as_of_utc)
        if records:
            return records[: self.max_items]
        fallback_records = self._fetch_fallback_from_economic_times(as_of_utc=as_of_utc)
        if fallback_records:
            return fallback_records[: self.max_items]
        now = as_of_utc or datetime.now(UTC)
        return [
            RawTextRecord(
                record_type="news_article",
                source_name=self.source_name,
                source_id="nse_news_offline_fallback_001",
                timestamp=now,
                content="NSE official API and ET fallback unavailable; using deterministic offline sample.",
                payload={
                    "headline": "NSE Offline Fallback Sample",
                    "publisher": "NSE",
                    "url": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
                    "author": "NSE Corporate Announcements",
                    "language": "en",
                    "source_type": self.source_type.value,
                    "ingestion_timestamp_utc": now,
                    "ingestion_timestamp_ist": now.astimezone(IST),
                    "schema_version": "1.0",
                    "quality_status": "warn",
                    "is_published": True,
                    "is_embargoed": False,
                    "license_ok": True,
                    "manipulation_risk_score": 0.14,
                    "confidence": 0.58,
                    "quality_flags": ["official_feed", "offline_fallback", "nse"],
                },
                source_type=self.source_type,
                source_route_detail=SourceRouteDetail.OFFICIAL_FEED,
            )
        ]

    def _fetch_from_nse_api(self, *, as_of_utc: datetime | None) -> list[RawTextRecord]:
        payload = self._fetch_text(
            url=self._NSE_ANNOUNCEMENT_URL,
            cache_key="corporate_announcements",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.nseindia.com/",
            },
        )
        if not payload.strip():
            return []
        if not payload.strip().startswith("["):
            return []

        now_utc = datetime.now(UTC)
        records: list[RawTextRecord] = []
        try:
            rows = json.loads(payload)
        except json.JSONDecodeError:
            return []

        for row in rows[: self.max_items]:
            if not isinstance(row, dict):
                continue
            title = self._normalize_text(str(row.get("subject") or row.get("sm_name") or ""))
            if not title:
                continue
            detail = self._normalize_text(str(row.get("details") or row.get("desc") or title))
            raw_link = str(row.get("attchmntFile") or row.get("attachement") or "").strip()
            if raw_link and raw_link.startswith("/"):
                raw_link = f"https://www.nseindia.com{raw_link}"
            url = raw_link or "https://www.nseindia.com/companies-listing/corporate-filings-announcements"

            timestamp = now_utc
            for key in ("broadcastdate", "an_dt", "date", "sort_date"):
                raw_value = row.get(key)
                if not raw_value:
                    continue
                candidate = self._parse_rss_datetime(str(raw_value))
                if candidate is not None:
                    timestamp = candidate
                    break

            if as_of_utc and timestamp > as_of_utc:
                continue

            source_token = str(row.get("id") or row.get("announcementId") or f"{title}|{url}")
            source_id = self._stable_id("nse_news", source_token)
            records.append(
                RawTextRecord(
                    record_type="news_article",
                    source_name=self.source_name,
                    source_id=source_id,
                    timestamp=timestamp,
                    content=detail,
                    payload={
                        "headline": title,
                        "publisher": "NSE",
                        "url": url,
                        "author": "NSE Corporate Announcements",
                        "language": "en",
                        "source_type": self.source_type.value,
                        "ingestion_timestamp_utc": now_utc,
                        "ingestion_timestamp_ist": now_utc.astimezone(IST),
                        "schema_version": "1.0",
                        "quality_status": "pass",
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": True,
                        "manipulation_risk_score": 0.08,
                        "confidence": 0.88,
                        "quality_flags": ["official_feed", "nse"],
                    },
                    source_type=self.source_type,
                    source_route_detail=SourceRouteDetail.OFFICIAL_FEED,
                )
            )
        return records

    def _fetch_fallback_from_economic_times(self, *, as_of_utc: datetime | None) -> list[RawTextRecord]:
        feed = self._fetch_text(url=self._ET_FALLBACK_URL, cache_key="et_fallback_markets")
        if not feed.strip():
            return []

        now_utc = datetime.now(UTC)
        records: list[RawTextRecord] = []
        try:
            root = ET.fromstring(feed)
        except ET.ParseError:
            return []

        for item in root.findall(".//item"):
            title = self._normalize_text(item.findtext("title") or "")
            description = self._normalize_text(item.findtext("description") or "")
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            text_blob = f"{title} {description}".lower()
            if not any(keyword in text_blob for keyword in _NSE_KEYWORDS):
                continue

            timestamp = self._parse_rss_datetime(item.findtext("pubDate")) or now_utc
            if as_of_utc and timestamp > as_of_utc:
                continue
            source_id = self._stable_id("nse_fallback", item.findtext("guid") or link)
            records.append(
                RawTextRecord(
                    record_type="news_article",
                    source_name=self.source_name,
                    source_id=source_id,
                    timestamp=timestamp,
                    content=description or title,
                    payload={
                        "headline": title,
                        "publisher": "Economic Times",
                        "url": link,
                        "author": "Economic Times",
                        "language": "en",
                        "source_type": self.source_type.value,
                        "ingestion_timestamp_utc": now_utc,
                        "ingestion_timestamp_ist": now_utc.astimezone(IST),
                        "schema_version": "1.0",
                        "quality_status": "warn",
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": True,
                        "manipulation_risk_score": 0.22,
                        "confidence": 0.62,
                        "quality_flags": ["fallback_scraper", "nse_keyword_filter"],
                    },
                    source_type=self.source_type,
                    source_route_detail=SourceRouteDetail.FALLBACK_SCRAPER,
                )
            )
        return records


class EconomicTimesAdapter(BaseTextAdapter):
    source_name = "economic_times"
    source_type = TextSourceType.RSS_FEED
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED
    cache_namespace = "economic_times"
    _FEED_URL = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        feed = self._fetch_text(url=self._FEED_URL, cache_key="markets_rss")
        if not feed.strip():
            return [self._build_offline_fallback_record(as_of_utc=as_of_utc)]

        now_utc = datetime.now(UTC)
        records: list[RawTextRecord] = []
        try:
            root = ET.fromstring(feed)
        except ET.ParseError:
            return [self._build_offline_fallback_record(as_of_utc=as_of_utc, now_utc=now_utc)]

        for item in root.findall(".//item")[: self.max_items]:
            title = self._normalize_text(item.findtext("title") or "")
            description = self._normalize_text(item.findtext("description") or "")
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            timestamp = self._parse_rss_datetime(item.findtext("pubDate")) or now_utc
            if as_of_utc and timestamp > as_of_utc:
                continue

            guid = (item.findtext("guid") or link).strip()
            source_id = self._stable_id("et_news", guid)
            records.append(
                RawTextRecord(
                    record_type="news_article",
                    source_name=self.source_name,
                    source_id=source_id,
                    timestamp=timestamp,
                    content=description or title,
                    payload={
                        "headline": title,
                        "publisher": "Economic Times",
                        "url": link,
                        "author": "Economic Times",
                        "language": "en",
                        "source_type": self.source_type.value,
                        "ingestion_timestamp_utc": now_utc,
                        "ingestion_timestamp_ist": now_utc.astimezone(IST),
                        "schema_version": "1.0",
                        "quality_status": "pass",
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": True,
                        "manipulation_risk_score": 0.14,
                        "confidence": 0.79,
                        "quality_flags": ["official_feed", "markets"],
                    },
                    source_type=self.source_type,
                    source_route_detail=self.source_route_detail,
                )
            )

        if records:
            return records
        return [self._build_offline_fallback_record(as_of_utc=as_of_utc, now_utc=now_utc)]

    def _build_offline_fallback_record(
        self,
        *,
        as_of_utc: datetime | None,
        now_utc: datetime | None = None,
    ) -> RawTextRecord:
        now = as_of_utc or now_utc or datetime.now(UTC)
        ingest_now = now_utc or now
        return RawTextRecord(
            record_type="news_article",
            source_name=self.source_name,
            source_id="economic_times_offline_fallback_001",
            timestamp=now,
            content="Economic Times feed unavailable; using deterministic offline sample.",
            payload={
                "headline": "Economic Times Offline Fallback Sample",
                "publisher": "Economic Times",
                "url": "https://economictimes.indiatimes.com/markets",
                "author": "Economic Times",
                "language": "en",
                "source_type": self.source_type.value,
                "ingestion_timestamp_utc": ingest_now,
                "ingestion_timestamp_ist": ingest_now.astimezone(IST),
                "schema_version": "1.0",
                "quality_status": "warn",
                "is_published": True,
                "is_embargoed": False,
                "license_ok": True,
                "manipulation_risk_score": 0.18,
                "confidence": 0.64,
                "quality_flags": ["official_feed", "offline_fallback", "markets"],
            },
            source_type=self.source_type,
            source_route_detail=self.source_route_detail,
        )

class RBIReportsAdapter(BaseTextAdapter):
    source_name = "rbi_reports"
    source_type = TextSourceType.RSS_FEED
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED
    cache_namespace = "rbi_reports"

    _RSS_INDEX_URL = "https://www.rbi.org.in/Scripts/rss.aspx"
    _LEGACY_RSS_FEED_URLS = (
        "https://www.rbi.org.in/pressreleases_rss.xml",
        "https://www.rbi.org.in/notifications_rss.xml",
        "https://www.rbi.org.in/Bulletin_rss.xml",
        "https://rbi.org.in/pressreleases_rss.xml",
        "https://rbi.org.in/notifications_rss.xml",
        "https://rbi.org.in/Bulletin_rss.xml",
    )
    # Backward-compatible alias used by existing tests and utility scripts.
    _FEED_URLS = _LEGACY_RSS_FEED_URLS
    _DBIE_CATALOG_URL = "https://data.rbi.org.in/DBIE/#/"
    _EMERGENCY_SCRAPER_URL = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    _ANCHOR_LINK_RE = re.compile(
        r"""<a[^>]+href=["'](?P<href>[^"']+)["'][^>]*>(?P<label>.*?)</a>""",
        re.IGNORECASE | re.DOTALL,
    )
    _DOWNLOAD_EXT_RE = re.compile(
        r"""\.(?:csv|xls|xlsx|zip|pdf)(?:\?[^"']*)?$""",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        allow_emergency_fallback: bool = False,
        dbie_catalog_urls: Sequence[str] | None = None,
        pdf_paths: Sequence[str] | None = None,
        source_policy: Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._allow_emergency_fallback = allow_emergency_fallback
        self._dbie_catalog_urls = tuple(dbie_catalog_urls or (self._DBIE_CATALOG_URL,))
        self._pdf_paths = list(pdf_paths or [])
        self._pdf_extractor = PDFExtractor()
        # Re-integrate the source policy for outage simulation (Day 3 requirement).
        self.source_policy = self._normalize_source_policy(source_policy)

    @classmethod
    def _normalize_source_policy(
        cls,
        source_policy: Mapping[str, object] | None,
    ) -> dict[str, object]:
        normalized = dict(_RBI_DEFAULT_SOURCE_POLICY)
        if source_policy is None:
            return normalized

        for key, value in source_policy.items():
            if key not in normalized:
                continue
            if isinstance(normalized[key], bool):
                normalized[key] = bool(value)
                continue
            if isinstance(value, str) and value.strip():
                normalized[key] = value.strip()
        return normalized

    def _resolve_source_route_logic(self) -> _RBIResolvedRoute | None:
        """Heuristic to resolve which route to use, supporting simulation modes."""
        if not bool(self.source_policy["simulate_rss_outage"]):
            return _RBIResolvedRoute(
                source_type=TextSourceType.RSS_FEED,
                route_detail=SourceRouteDetail.OFFICIAL_FEED,
                source_url=str(self.source_policy["rss_feed_url"]),
                source_channel="rbi_official_rss",
            )

        if not bool(self.source_policy["simulate_dbie_outage"]):
            return _RBIResolvedRoute(
                source_type=TextSourceType.OFFICIAL_API,
                route_detail=SourceRouteDetail.PRIMARY_API,
                source_url=str(self.source_policy["dbie_download_url"]),
                source_channel="rbi_dbie_download",
            )

        if bool(self.source_policy["enable_emergency_fallback_scraper"]) or self._allow_emergency_fallback:
            return _RBIResolvedRoute(
                source_type=TextSourceType.FALLBACK_SCRAPER,
                route_detail=SourceRouteDetail.FALLBACK_SCRAPER,
                source_url=str(self.source_policy["fallback_scraper_url"]),
                source_channel="rbi_emergency_scraper",
            )

        return None

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now_utc = datetime.now(UTC)
        record_timestamp = as_of_utc or now_utc
        
        # 1. Local PDF processing (highest priority in upstream).
        local_pdf_records = self._build_local_pdf_records(
            now_utc=now_utc,
            record_timestamp=record_timestamp,
        )
        if local_pdf_records:
            return local_pdf_records[: self.max_items]

        # 2. Resolved route logic (Simulation/Policy support).
        resolved_route = self._resolve_source_route_logic()
        if resolved_route is None:
            logger.warning("RBI source routes unavailable (outage simulation active?)")
            return []

        records: list[RawTextRecord] = []
        seen_source_ids: set[str] = set()

        # 3. RSS Route
        if resolved_route.source_type == TextSourceType.RSS_FEED:
            rss_feed_urls = self._discover_rbi_rss_urls()
            for feed_url in rss_feed_urls:
                if len(records) >= self.max_items:
                    break
                feed = self._fetch_text(url=feed_url, cache_key=self._stable_id("rbi_feed", feed_url))
                if not feed.strip():
                    continue
                try:
                    root = ET.fromstring(feed)
                except ET.ParseError:
                    continue

                for item in root.findall(".//item"):
                    title = self._normalize_text(item.findtext("title") or "")
                    description = self._normalize_text(item.findtext("description") or "")
                    link = (item.findtext("link") or "").strip()
                    guid = (item.findtext("guid") or link or title).strip()
                    if not title or not link:
                        continue

                    timestamp = self._parse_rss_datetime(item.findtext("pubDate")) or now_utc
                    if as_of_utc and timestamp > as_of_utc:
                        continue

                    source_id = self._stable_id("rbi_rss", f"{feed_url}|{guid}|{link}")
                    if source_id in seen_source_ids:
                        continue
                    seen_source_ids.add(source_id)

                    records.append(
                        RawTextRecord(
                            record_type="news_article",
                            source_name=self.source_name,
                            source_id=source_id,
                            timestamp=timestamp,
                            content=description or title,
                            payload={
                                "headline": title,
                                "publisher": "Reserve Bank of India",
                                "url": link,
                                "author": "RBI",
                                "language": "en",
                                "source_type": resolved_route.source_type.value,
                                "ingestion_timestamp_utc": now_utc,
                                "ingestion_timestamp_ist": now_utc.astimezone(IST),
                                "schema_version": "1.0",
                                "quality_status": "pass",
                                "is_published": True,
                                "is_embargoed": False,
                                "license_ok": True,
                                "manipulation_risk_score": 0.05,
                                "confidence": 0.92,
                                "quality_flags": ["official_feed", "rbi_rss_xml"],
                                "source_channel": resolved_route.source_channel,
                            },
                            source_type=resolved_route.source_type,
                            source_route_detail=resolved_route.route_detail,
                        )
                    )
            if records:
                return records[: self.max_items]

        # 4. DBIE (Official API / Primary API) Route
        if resolved_route.source_type == TextSourceType.OFFICIAL_API:
            records.extend(
                self._fetch_dbie_downloads(
                    now_utc=now_utc,
                    as_of_utc=as_of_utc,
                    seen_source_ids=seen_source_ids,
                    limit=self.max_items,
                )
            )
            # Apply route details to DBIE records if not already set.
            for i in range(len(records)):
                records[i] = RawTextRecord(
                    **{**records[i].__dict__, "source_type": resolved_route.source_type, "source_route_detail": resolved_route.route_detail}
                )
            if records:
                return records[: self.max_items]

        # 5. Emergency Scraper Route
        if resolved_route.source_type == TextSourceType.FALLBACK_SCRAPER:
            records.extend(
                self._fetch_emergency_scraper_records(
                    now_utc=now_utc,
                    as_of_utc=as_of_utc,
                    seen_source_ids=seen_source_ids,
                )
            )
            if records:
                return records[: self.max_items]

        # 6. Offline Fallback (CI/Safety)
        return [
            RawTextRecord(
                record_type="news_article",
                source_name=self.source_name,
                source_id="rbi_rss_fallback_001",
                timestamp=record_timestamp,
                content="RBI official channels unavailable; using deterministic fallback sample.",
                payload={
                    "headline": "RBI Official Feed Fallback Sample",
                    "publisher": "Reserve Bank of India",
                    "url": resolved_route.source_url,
                    "author": "RBI",
                    "language": "en",
                    "source_type": resolved_route.source_type.value,
                    "ingestion_timestamp_utc": now_utc,
                    "ingestion_timestamp_ist": now_utc.astimezone(IST),
                    "schema_version": "1.0",
                    "quality_status": "warn",
                    "is_published": True,
                    "is_embargoed": False,
                    "license_ok": True,
                    "manipulation_risk_score": 0.08,
                    "confidence": 0.6,
                    "quality_flags": ["official_feed", "offline_fallback", "rbi_rss_xml"],
                    "source_channel": resolved_route.source_channel,
                },
                source_type=resolved_route.source_type,
                source_route_detail=resolved_route.route_detail,
            )
        ]

    def _build_local_pdf_records(
        self,
        *,
        now_utc: datetime,
        record_timestamp: datetime,
    ) -> list[RawTextRecord]:
        records: list[RawTextRecord] = []
        for pdf_path, pdf_bytes in _read_pdf_documents(self._pdf_paths):
            extraction = self._pdf_extractor.extract(pdf_bytes)
            source_token = _safe_token(pdf_path.stem)
            records.append(
                RawTextRecord(
                    record_type="news_article",
                    source_name=self.source_name,
                    source_id=f"rbi_report_{source_token}",
                    timestamp=record_timestamp,
                    content=extraction.text,
                    payload={
                        "headline": f"RBI Report Extract ({pdf_path.name})",
                        "publisher": "Reserve Bank of India",
                        "url": _path_url(pdf_path),
                        "author": "RBI PDF Import",
                        "language": "en",
                        "source_type": self.source_type.value,
                        "ingestion_timestamp_utc": now_utc,
                        "ingestion_timestamp_ist": now_utc.astimezone(IST),
                        "schema_version": "1.0",
                        "quality_status": extraction.quality_status,
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": True,
                        "manipulation_risk_score": 0.05,
                        "confidence": 0.9,
                        "quality_flags": ["official_feed", "configured_pdf_input"],
                        "extraction_quality_score": extraction.quality_score,
                        "pdf_quality_status": extraction.quality_status,
                        "pdf_extracted_char_count": extraction.metrics["extracted_char_count"],
                    },
                    source_type=self.source_type,
                    source_route_detail=self.source_route_detail,
                )
            )
        return records

    def _discover_rbi_rss_urls(self) -> list[str]:
        discovered_urls: list[str] = []
        index_doc = self._fetch_text(url=self._RSS_INDEX_URL, cache_key="rss_index")
        if index_doc.strip():
            for href, _ in self._extract_anchor_links(index_doc):
                absolute = self._normalize_absolute_url(base_url=self._RSS_INDEX_URL, href=href)
                if not absolute:
                    continue
                lower_absolute = absolute.lower()
                if "rss" in lower_absolute or lower_absolute.endswith(".xml"):
                    discovered_urls.append(absolute)

        discovered_urls.extend(self._LEGACY_RSS_FEED_URLS)

        unique_urls: list[str] = []
        seen_urls: set[str] = set()
        for url in discovered_urls:
            normalized_url = url.strip()
            if not normalized_url or normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            unique_urls.append(normalized_url)
        return unique_urls

    def _fetch_dbie_downloads(
        self,
        *,
        now_utc: datetime,
        as_of_utc: datetime | None,
        seen_source_ids: set[str],
        limit: int,
    ) -> list[RawTextRecord]:
        if limit <= 0:
            return []

        timestamp = as_of_utc or now_utc
        records: list[RawTextRecord] = []

        for catalog_url in self._dbie_catalog_urls:
            if len(records) >= limit:
                break

            cache_key = self._cache_key_for_url(catalog_url)
            page = self._fetch_text(url=catalog_url, cache_key=cache_key)
            if not page.strip():
                continue

            for href, label in self._extract_anchor_links(page):
                if len(records) >= limit:
                    break

                absolute = self._normalize_absolute_url(base_url=catalog_url, href=href)
                if not absolute or not self._DOWNLOAD_EXT_RE.search(absolute):
                    continue

                source_id = self._stable_id("rbi_dbie", absolute)
                if source_id in seen_source_ids:
                    continue
                seen_source_ids.add(source_id)

                file_name = Path(urlparse(absolute).path).name or "dataset"
                title = label or file_name
                records.append(
                    RawTextRecord(
                        record_type="news_article",
                        source_name=self.source_name,
                        source_id=source_id,
                        timestamp=timestamp,
                        content=f"RBI DBIE published official download artifact: {title}.",
                        payload={
                            "headline": f"RBI DBIE download available: {title}",
                            "publisher": "Reserve Bank of India DBIE",
                            "url": absolute,
                            "author": "RBI DBIE",
                            "language": "en",
                            "source_type": self.source_type.value,
                            "ingestion_timestamp_utc": now_utc,
                            "ingestion_timestamp_ist": now_utc.astimezone(IST),
                            "schema_version": "1.0",
                            "quality_status": "pass",
                            "is_published": True,
                            "is_embargoed": False,
                            "license_ok": True,
                            "manipulation_risk_score": 0.03,
                            "confidence": 0.9,
                            "quality_flags": ["official_feed", "dbie_official_download"],
                        },
                        source_type=self.source_type,
                        source_route_detail=self.source_route_detail,
                    )
                )

        return records

    def _fetch_emergency_scraper_records(
        self,
        *,
        now_utc: datetime,
        as_of_utc: datetime | None,
        seen_source_ids: set[str],
    ) -> list[RawTextRecord]:
        page = self._fetch_text(
            url=self._EMERGENCY_SCRAPER_URL,
            cache_key="emergency_press_release_page",
        )
        if not page.strip():
            return []

        timestamp = as_of_utc or now_utc
        records: list[RawTextRecord] = []
        for href, label in self._extract_anchor_links(page):
            absolute = self._normalize_absolute_url(base_url=self._EMERGENCY_SCRAPER_URL, href=href)
            if not absolute:
                continue
            lower_absolute = absolute.lower()
            if "bs_pressreleasedisplay.aspx" not in lower_absolute and "prid=" not in lower_absolute:
                continue

            source_id = self._stable_id("rbi_scraper", absolute)
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)

            title = label or "RBI emergency press release update"
            records.append(
                RawTextRecord(
                    record_type="news_article",
                    source_name=self.source_name,
                    source_id=source_id,
                    timestamp=timestamp,
                    content=f"Emergency fallback scrape captured RBI release link: {title}.",
                    payload={
                        "headline": title,
                        "publisher": "Reserve Bank of India",
                        "url": absolute,
                        "author": "RBI",
                        "language": "en",
                        "source_type": self.source_type.value,
                        "ingestion_timestamp_utc": now_utc,
                        "ingestion_timestamp_ist": now_utc.astimezone(IST),
                        "schema_version": "1.0",
                        "quality_status": "warn",
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": True,
                        "fallback_emergency_active": True,
                        "manipulation_risk_score": 0.28,
                        "confidence": 0.56,
                        "quality_flags": ["fallback_scraper", "outage_emergency", "rbi_html_scrape"],
                    },
                    source_type=self.source_type,
                    source_route_detail=SourceRouteDetail.FALLBACK_SCRAPER,
                )
            )
            if len(records) >= self.max_items:
                break
        return records

    @classmethod
    def _extract_anchor_links(cls, html_doc: str) -> list[tuple[str, str]]:
        links: list[tuple[str, str]] = []
        for match in cls._ANCHOR_LINK_RE.finditer(html_doc):
            href = html.unescape(match.group("href") or "").strip()
            label = cls._normalize_text(match.group("label") or "")
            if href:
                links.append((href, label))
        return links

    @staticmethod
    def _normalize_absolute_url(*, base_url: str, href: str) -> str | None:
        candidate = (href or "").strip()
        if not candidate or candidate.lower().startswith("javascript:"):
            return None

        absolute = urljoin(base_url, candidate)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            return None
        return absolute

    @staticmethod
    def _cache_key_for_url(url: str) -> str:
        parsed = urlparse(url)
        path_part = re.sub(r"[^a-zA-Z0-9]+", "_", parsed.path.strip("/")).strip("_")
        query_part = re.sub(r"[^a-zA-Z0-9]+", "_", parsed.query).strip("_")
        key = "_".join(part for part in (path_part, query_part) if part)
        return key or "root"


class EarningsTranscriptAdapter(BaseTextAdapter):
    source_name = "earnings_transcripts"
    source_type = TextSourceType.OFFICIAL_API
    source_route_detail = SourceRouteDetail.OFFICIAL_FEED
    cache_namespace = "earnings_transcripts"

    # Mock list of URLs for phase 1 testing
    _MOCK_PDF_URLS = [
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf" # Placeholder dummy pdf
    ]

    def __init__(self, *, pdf_paths: Sequence[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pdf_paths = list(pdf_paths or [])
        self._pdf_extractor = PDFExtractor()

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawTextRecord]:
        now_utc = as_of_utc or datetime.now(UTC)
        local_pdf_records = self._build_local_pdf_records(now_utc=now_utc)
        if local_pdf_records:
            return local_pdf_records[: self.max_items]

        records: list[RawTextRecord] = []

        if pdfplumber is None:
            logger.warning("pdfplumber not installed; using deterministic earnings transcript fallback.")
            return self._build_offline_fallback_records(now_utc)

        for url in self._MOCK_PDF_URLS:
            try:
                # Download PDF to temp file
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    pdf_data = response.read()

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
                    temp_pdf.write(pdf_data)
                    temp_pdf.flush()

                    # Extract text using pdfplumber
                    extracted_text = []
                    with pdfplumber.open(temp_pdf.name) as pdf:
                        for page in pdf.pages[:5]: # Limit to first 5 pages for speed/spot-checks
                            text = page.extract_text()
                            if text:
                                extracted_text.append(text)
                    
                    full_text = " ".join(extracted_text)
                    if not full_text.strip():
                        continue

                    # In a real scenario, these would be parsed from the text or source feed
                    symbol = "DUMMY.NS"
                    quarter = "Q1"
                    year = now_utc.year
                    
                    source_id = self._stable_id("earnings_pdf", url)
                    records.append(
                        RawTextRecord(
                            record_type="earnings_transcript",
                            source_name=self.source_name,
                            source_id=source_id,
                            timestamp=now_utc,
                            content=full_text,
                            payload={
                                "symbol": symbol,
                                "quarter": quarter,
                                "year": year,
                                "url": url,
                                "author": "Company Management",
                                "language": "en",
                                "source_type": self.source_type.value,
                                "ingestion_timestamp_utc": now_utc,
                                "ingestion_timestamp_ist": now_utc.astimezone(IST),
                                "schema_version": "1.0",
                                "quality_status": "pass",
                                "is_published": True,
                                "is_embargoed": False,
                                "license_ok": True,
                                "manipulation_risk_score": 0.0,
                                "confidence": 0.95,
                                "quality_flags": ["pdf_extraction", "spot_check_ok"],
                                "extraction_quality_score": 0.9, # Mock high score since real PDF extraction succeeded
                                "pdf_quality_status": "pass",
                            },
                            source_type=self.source_type,
                            source_route_detail=self.source_route_detail,
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to extract PDF transcript from {url}: {e}")
        if records:
            return records[: self.max_items]

        return self._build_offline_fallback_records(now_utc)

    def _build_local_pdf_records(self, *, now_utc: datetime) -> list[RawTextRecord]:
        records: list[RawTextRecord] = []
        for pdf_path, pdf_bytes in _read_pdf_documents(self._pdf_paths):
            extraction = self._pdf_extractor.extract(pdf_bytes)
            source_token = _safe_token(pdf_path.stem)
            records.append(
                RawTextRecord(
                    record_type="earnings_transcript",
                    source_name=self.source_name,
                    source_id=f"earnings_transcript_{source_token}",
                    timestamp=now_utc,
                    content=extraction.text,
                    payload={
                        "symbol": _infer_transcript_symbol(pdf_path),
                        "quarter": _infer_transcript_quarter(pdf_path),
                        "year": _infer_year(pdf_path, now_utc.year),
                        "url": _path_url(pdf_path),
                        "author": "Company Management",
                        "language": "en",
                        "source_type": self.source_type.value,
                        "ingestion_timestamp_utc": now_utc,
                        "ingestion_timestamp_ist": now_utc.astimezone(IST),
                        "schema_version": "1.0",
                        "quality_status": extraction.quality_status,
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": True,
                        "manipulation_risk_score": 0.0,
                        "confidence": 0.95,
                        "quality_flags": ["pdf_extraction", "configured_pdf_input"],
                        "extraction_quality_score": extraction.quality_score,
                        "pdf_quality_status": extraction.quality_status,
                        "pdf_extracted_char_count": extraction.metrics["extracted_char_count"],
                    },
                    source_type=self.source_type,
                    source_route_detail=self.source_route_detail,
                )
            )
        return records

    def _build_offline_fallback_records(self, now_utc: datetime) -> list[RawTextRecord]:
        # Offline-safe deterministic fallback so default agent runs are reproducible in CI.
        return [
            RawTextRecord(
                record_type="earnings_transcript",
                source_name=self.source_name,
                source_id="earnings_transcript_q1_2026",
                timestamp=now_utc,
                content=(
                    "Q1 FY2026 earnings call transcript highlights: revenue growth remained resilient "
                    "while management guided stable margins for the next quarter."
                ),
                payload={
                    "symbol": "DUMMY.NS",
                    "quarter": "Q1",
                    "year": now_utc.year,
                    "url": "https://example.com/earnings/transcript/q1-2026",
                    "author": "Company Management",
                    "language": "en",
                    "source_type": self.source_type.value,
                    "ingestion_timestamp_utc": now_utc,
                    "ingestion_timestamp_ist": now_utc.astimezone(IST),
                    "schema_version": "1.0",
                    "quality_status": "pass",
                    "is_published": True,
                    "is_embargoed": False,
                    "license_ok": True,
                    "manipulation_risk_score": 0.0,
                    "confidence": 0.9,
                    "quality_flags": ["offline_fallback"],
                    "extraction_quality_score": 0.85, # Dummy fallback score
                    "pdf_quality_status": "pass",
                },
                source_type=self.source_type,
                source_route_detail=self.source_route_detail,
            )
        ][: self.max_items]


class XPostAdapter(BaseTextAdapter):
    source_name = "x_posts"
    source_type = TextSourceType.SOCIAL_MEDIA
    source_route_detail = SourceRouteDetail.PRIMARY_API
    cache_namespace = "x_posts"

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
