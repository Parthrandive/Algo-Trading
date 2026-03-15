"""
Scraper for fetching real Foreign Exchange Reserves data from the RBI website.

The RBI publishes weekly FX reserves in the Weekly Statistical Supplement (WSS).
The data is published as a press release with title:
    "Reserve Bank of India – Bulletin Weekly Statistical Supplement – Extract"

The detail page contains Table 2: "Foreign Exchange Reserves" with a
"Total Reserves" row. The value is in US$ Mn (millions) in the third cell.

Source: https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx
"""

import logging
import re
import urllib.request
from datetime import UTC, datetime
from typing import Any

from src.agents.macro.client import DateRange

logger = logging.getLogger(__name__)

_RBI_PR_URL = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.rbi.org.in/",
}

# Pattern to find WSS press release links on the listing page.
# Example: <a class='link2' href='BS_PressReleaseDisplay.aspx?prid=62341'>
#   Reserve Bank of India – Bulletin Weekly Statistical Supplement – Extract</a>
_WSS_LINK_RE = re.compile(
    r"<a[^>]+href=['\"]([^'\"]*prid=\d+)['\"][^>]*>"
    r"(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)

# Pattern to find the "Total Reserves" row in the WSS detail page table.
# The row has cells like: <td>1 Total Reserves</td><td>6627548</td><td>728494</td>...
# We want the US$ Mn value (third cell = index 2).
_TOTAL_RESERVES_ROW_RE = re.compile(
    r"(?:1\s+)?Total\s+Reserves"          # Row label (may start with "1")
    r".*?"                                  # Stuff between
    r"<td[^>]*>\s*([\d,]+)\s*</td>"        # First numeric cell (₹ Cr.)
    r"\s*<td[^>]*>\s*([\d,]+)\s*</td>",    # Second numeric cell (US$ Mn.)
    re.IGNORECASE | re.DOTALL,
)

# Date pattern: "As on February 28, 2026" or similar
_AS_ON_DATE_RE = re.compile(
    r"[Aa]s\s+on\s+(\w+\s+\d{1,2},?\s+\d{4})",
)

# Alternative: "for the week ended February 28, 2026"
_WEEK_ENDED_DATE_RE = re.compile(
    r"week\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def _parse_rbi_date(text: str) -> str | None:
    """Extract a date from RBI-style text like 'February 28, 2026' → '2026-02-28'."""
    # Clean commas
    text = text.replace(",", "").strip()
    parts = text.split()
    if len(parts) != 3:
        return None

    month_str, day_str, year_str = parts
    month = _MONTH_NAMES.get(month_str.lower())
    if month is None:
        return None
    try:
        day = int(day_str)
        year = int(year_str)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _fetch_html(url: str, timeout: int = 30) -> str:
    """Fetch a URL and return the decoded HTML, with retry."""
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset)
        except Exception as e:
            last_err = e
            logger.warning("Fetch attempt %d for %s failed: %s", attempt + 1, url, e)
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts: {last_err}")


def fetch_real_fx_reserves(date_range: DateRange) -> dict[str, Any]:
    """
    Fetches real FX Reserves data from the RBI website.

    Scrapes the RBI Press Releases page for the latest
    "Weekly Statistical Supplement – Extract" link, then parses the
    detail page's "Total Reserves" table row for the US$ Mn value.

    Args:
        date_range: The date range to query. Since FX reserves are weekly,
                    we return the latest available data point.

    Returns:
        dict: A payload parsable by FXReservesParser, e.g.:
            {"date": "2026-02-27", "value": 728.494}

    Raises:
        RuntimeError: If scraping fails at any step.
    """
    logger.info("Fetching real FX Reserves from RBI press releases...")

    # Step 1: Fetch the press release listing page
    try:
        listing_html = _fetch_html(_RBI_PR_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch RBI press release listing: {e}") from e

    # Step 2: Find the latest WSS press release link
    wss_prid = None
    for href, label_html in _WSS_LINK_RE.findall(listing_html):
        # Strip HTML tags from label
        label_text = re.sub(r"<[^>]+>", "", label_html).strip()
        if "weekly statistical supplement" in label_text.lower():
            # Extract prid from href
            prid_match = re.search(r"prid=(\d+)", href)
            if prid_match:
                wss_prid = prid_match.group(1)
                logger.info(
                    "Found WSS press release: prid=%s label='%s'",
                    wss_prid, label_text[:80],
                )
                break

    if not wss_prid:
        raise RuntimeError(
            "Could not find 'Weekly Statistical Supplement' link on RBI press release page."
        )

    # Step 3: Fetch the WSS detail page
    detail_url = f"https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid={wss_prid}"
    try:
        detail_html = _fetch_html(detail_url)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch WSS detail page (prid={wss_prid}): {e}") from e

    # Step 4: Extract the "Total Reserves" value in US$ Mn
    row_match = _TOTAL_RESERVES_ROW_RE.search(detail_html)
    if not row_match:
        raise RuntimeError(
            "Could not find 'Total Reserves' row in WSS detail page."
        )

    # Group 1 = ₹ Cr. value, Group 2 = US$ Mn. value
    usd_mn_str = row_match.group(2).replace(",", "")
    try:
        usd_mn = float(usd_mn_str)
    except ValueError as e:
        raise RuntimeError(f"Could not parse US$ Mn value '{usd_mn_str}': {e}") from e

    # Convert from millions to billions
    value_usd_bn = round(usd_mn / 1000.0, 3)

    # Step 5: Extract observation date ("As on ..." or "week ended ...")
    observation_date = None
    for pattern in (_AS_ON_DATE_RE, _WEEK_ENDED_DATE_RE):
        date_match = pattern.search(detail_html)
        if date_match:
            observation_date = _parse_rbi_date(date_match.group(1))
            if observation_date:
                break

    # Fall back to the pipeline's date range end if we can't extract the date
    if not observation_date:
        observation_date = date_range.end.isoformat()
        logger.warning(
            "Could not extract observation date from WSS page; using date_range.end=%s",
            observation_date,
        )

    payload = {
        "date": observation_date,
        "value": value_usd_bn,
    }
    logger.info("Successfully scraped FX Reserves: %s", payload)
    return payload
