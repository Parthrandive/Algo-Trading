from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.agents.textual.adapters import EconomicTimesAdapter, NSENewsAdapter, RBIReportsAdapter
from src.schemas.text_sidecar import SourceRouteDetail


def _workspace_tmp_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "test_tmp" / f"adapters_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _http_get_from_map(payload_map: dict[str, str]):
    def _getter(url: str, headers: dict[str, str] | None = None) -> str:
        _ = headers
        if url not in payload_map:
            raise ValueError(f"unexpected url: {url}")
        return payload_map[url]

    return _getter


def test_economic_times_adapter_parses_rss_item():
    rss_payload = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title><![CDATA[Test market headline]]></title>
      <description><![CDATA[<p>Market moved higher after policy cues.</p>]]></description>
      <link>https://example.com/market/headline</link>
      <guid>guid-1</guid>
      <pubDate>Thu, 05 Mar 2026 10:00:00 +0530</pubDate>
    </item>
  </channel>
</rss>
"""

    adapter = EconomicTimesAdapter(
        max_items=5,
        http_get=_http_get_from_map({EconomicTimesAdapter._FEED_URL: rss_payload}),
        cache_root=_workspace_tmp_dir() / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.record_type == "news_article"
    assert record.source_name == "economic_times"
    assert record.payload["headline"] == "Test market headline"
    assert record.payload["publisher"] == "Economic Times"
    assert record.payload["url"] == "https://example.com/market/headline"


def test_rbi_reports_adapter_parses_rss_item():
    feed_payload = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title><![CDATA[RBI publishes policy update]]></title>
      <description><![CDATA[<p>Policy update paragraph one.</p>]]></description>
      <link>https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=62334</link>
      <guid>rbi-guid-62334</guid>
      <pubDate>Thu, 05 Mar 2026 10:00:00 +0530</pubDate>
    </item>
  </channel>
</rss>
"""
    adapter = RBIReportsAdapter(
        max_items=1,
        http_get=_http_get_from_map(
            {
                RBIReportsAdapter._FEED_URLS[0]: feed_payload,
            }
        ),
        cache_root=_workspace_tmp_dir() / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_name == "rbi_reports"
    assert record.payload["headline"] == "RBI publishes policy update"
    assert record.payload["publisher"] == "Reserve Bank of India"
    assert record.payload["url"] == "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=62334"
    assert "Policy update paragraph one." in record.content
    assert record.source_route_detail == SourceRouteDetail.OFFICIAL_FEED
    assert "rbi_rss_xml" in record.payload["quality_flags"]


def test_rbi_reports_adapter_parses_dbie_download_item():
    dbie_page = """
<html>
  <body>
    <a href="/dbie/downloads/fx_reserves_2026_03_05.csv">FX Reserves Weekly</a>
  </body>
</html>
"""
    payload_map = {
        RBIReportsAdapter._RSS_INDEX_URL: "",
        RBIReportsAdapter._DBIE_CATALOG_URL: dbie_page,
    }
    for url in RBIReportsAdapter._FEED_URLS:
        payload_map[url] = ""

    adapter = RBIReportsAdapter(
        max_items=5,
        http_get=_http_get_from_map(payload_map),
        cache_root=_workspace_tmp_dir() / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_name == "rbi_reports"
    assert record.source_route_detail == SourceRouteDetail.OFFICIAL_FEED
    assert record.payload["publisher"] == "Reserve Bank of India DBIE"
    assert record.payload["url"].endswith("fx_reserves_2026_03_05.csv")
    assert "dbie_official_download" in record.payload["quality_flags"]


def test_rbi_reports_adapter_emergency_scraper_route():
    emergency_page = """
<html>
  <body>
    <a href="/Scripts/BS_PressReleaseDisplay.aspx?prid=70001">Emergency RBI release</a>
  </body>
</html>
"""
    payload_map = {
        RBIReportsAdapter._RSS_INDEX_URL: "",
        RBIReportsAdapter._DBIE_CATALOG_URL: "",
        RBIReportsAdapter._EMERGENCY_SCRAPER_URL: emergency_page,
    }
    for url in RBIReportsAdapter._FEED_URLS:
        payload_map[url] = ""

    adapter = RBIReportsAdapter(
        max_items=5,
        allow_emergency_fallback=True,
        http_get=_http_get_from_map(payload_map),
        cache_root=_workspace_tmp_dir() / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_route_detail == SourceRouteDetail.FALLBACK_SCRAPER
    assert record.payload["fallback_emergency_active"] is True
    assert "outage_emergency" in record.payload["quality_flags"]


def test_nse_adapter_uses_et_fallback_when_nse_unavailable():
    fallback_rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title><![CDATA[Nifty jumps as banking stocks rally]]></title>
      <description><![CDATA[Sentiment improved across NSE names.]]></description>
      <link>https://example.com/nse-fallback</link>
      <guid>nse-guid-1</guid>
      <pubDate>Thu, 05 Mar 2026 10:00:00 +0530</pubDate>
    </item>
  </channel>
</rss>
"""

    payload_map = {
        NSENewsAdapter._NSE_ANNOUNCEMENT_URL: "",
        NSENewsAdapter._ET_FALLBACK_URL: fallback_rss,
    }
    adapter = NSENewsAdapter(
        max_items=5,
        http_get=_http_get_from_map(payload_map),
        cache_root=_workspace_tmp_dir() / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_name == "nse_news"
    assert record.source_route_detail == SourceRouteDetail.FALLBACK_SCRAPER
    assert record.payload["quality_status"] == "warn"
