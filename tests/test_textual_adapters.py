from datetime import UTC, datetime
from pathlib import Path

from src.agents.textual.adapters import EconomicTimesAdapter, NSENewsAdapter, RBIReportsAdapter
from src.schemas.text_sidecar import SourceRouteDetail


def _http_get_from_map(payload_map: dict[str, str]):
    def _getter(url: str, headers: dict[str, str] | None = None) -> str:
        _ = headers
        if url not in payload_map:
            raise ValueError(f"unexpected url: {url}")
        return payload_map[url]

    return _getter


def test_economic_times_adapter_parses_rss_item(tmp_path: Path):
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
        cache_root=tmp_path / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.record_type == "news_article"
    assert record.source_name == "economic_times"
    assert record.payload["headline"] == "Test market headline"
    assert record.payload["publisher"] == "Economic Times"
    assert record.payload["url"] == "https://example.com/market/headline"


def test_rbi_reports_adapter_parses_press_release_html(tmp_path: Path):
    listing_payload = """
<html><body>
<a href="BS_PressReleaseDisplay.aspx?prid=62334">Release</a>
</body></html>
"""
    detail_payload = """
<html><body>
<table class="tablebg">
  <tr><td class="tableheader"><b> Date : Mar 05, 2026</b></td></tr>
  <tr><td align="center" class="tableheader"><b>RBI publishes policy update</b></td></tr>
  <tr class="tablecontent1">
    <td>
      <table><tr><td>
        <p>Policy update paragraph one.</p>
        <p>Policy update paragraph two.</p>
        <p class="head">Press Release: 2025-2026/2218</p>
      </td></tr></table>
    </td>
  </tr>
</table>
</body></html>
"""
    detail_url = RBIReportsAdapter._DETAIL_URL_TEMPLATE.format(prid="62334")
    adapter = RBIReportsAdapter(
        max_items=5,
        http_get=_http_get_from_map(
            {
                RBIReportsAdapter._LISTING_URL: listing_payload,
                detail_url: detail_payload,
            }
        ),
        cache_root=tmp_path / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_name == "rbi_reports"
    assert record.payload["headline"] == "RBI publishes policy update"
    assert record.payload["publisher"] == "Reserve Bank of India"
    assert record.payload["url"] == detail_url
    assert "Policy update paragraph one." in record.content


def test_nse_adapter_uses_et_fallback_when_nse_unavailable(tmp_path: Path):
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
        cache_root=tmp_path / "cache",
    )
    records = adapter.fetch(as_of_utc=datetime(2026, 3, 5, 23, 59, tzinfo=UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_name == "nse_news"
    assert record.source_route_detail == SourceRouteDetail.FALLBACK_SCRAPER
    assert record.payload["quality_status"] == "warn"
