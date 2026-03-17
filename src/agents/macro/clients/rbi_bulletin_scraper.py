"""
Scraper for fetching real RBI Bulletin publications.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from typing import Any

from src.agents.macro.client import DateRange

logger = logging.getLogger(__name__)

_RBI_BULLETIN_URL = "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx"

_HIDDEN_INPUT_RE = re.compile(
    r"<input[^>]+type=\"hidden\"[^>]+name=\"(?P<name>[^\"]+)\"[^>]+value=\"(?P<value>[^\"]*)\"",
    re.IGNORECASE,
)
_YEAR_MONTH_RE = re.compile(r"GetYearMonth\(\"(?P<year>\d{4})\",\"(?P<month>\d{1,2})\"\)")
_DATE_HEADER_RE = re.compile(r"Date\s*&nbsp;\s*:\s*&nbsp;\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4})", re.IGNORECASE)
_BULLETIN_TITLE_RE = re.compile(r"Reserve\s+Bank\s+of\s+India\s+Bulletin\s*-\s*([A-Za-z]+\s+\d{4})", re.IGNORECASE)


def _extract_hidden_fields(html: str) -> dict[str, str]:
    return {m.group("name"): m.group("value") for m in _HIDDEN_INPUT_RE.finditer(html)}


def _extract_year_month_options(html: str) -> list[tuple[int, int]]:
    raw_options: set[tuple[int, int]] = set()
    for match in _YEAR_MONTH_RE.finditer(html):
        year = int(match.group("year"))
        month = int(match.group("month"))
        raw_options.add((year, month))

    years_with_all = {year for year, month in raw_options if month == 0}
    options: set[tuple[int, int]] = set()
    for year, month in raw_options:
        if year in years_with_all:
            options.add((year, 0))
        elif month != 0:
            options.add((year, month))
    return sorted(options)


def _parse_bulletin_publications(html: str) -> list[dict[str, str]]:
    dates = _DATE_HEADER_RE.findall(html)
    titles = _BULLETIN_TITLE_RE.findall(html)

    publications: list[dict[str, str]] = []
    for raw_date, title in zip(dates, titles):
        parsed_date = datetime.strptime(raw_date.strip(), "%b %d, %Y").replace(tzinfo=UTC).date().isoformat()
        publications.append(
            {
                "date": parsed_date,
                "title": f"Reserve Bank of India Bulletin - {title.strip()}",
            }
        )
    return publications


def _post_year_month(html: str, year: int, month: int) -> str:
    state = _extract_hidden_fields(html)
    payload = {
        "__EVENTTARGET": state.get("__EVENTTARGET", ""),
        "__EVENTARGUMENT": state.get("__EVENTARGUMENT", ""),
        "__VIEWSTATE": state.get("__VIEWSTATE", ""),
        "__VIEWSTATEGENERATOR": state.get("__VIEWSTATEGENERATOR", ""),
        "__EVENTVALIDATION": state.get("__EVENTVALIDATION", ""),
        "hdnYear": str(year),
        "hdnMonth": str(month),
        "hndSearch_Flag": "1",
        "ddlSubSection": "0",
        "UsrFontCntr$btn": "",
    }
    encoded = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        _RBI_BULLETIN_URL,
        data=encoded,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": _RBI_BULLETIN_URL,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310 - trusted HTTPS endpoint
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")

def fetch_real_rbi_bulletin(date_range: DateRange) -> dict[str, Any]:
    """
    Fetches real RBI Bulletin titles from the RBI website.
    
    Args:
        date_range: The date range to query.
    Returns:
        dict: A payload parsable by RBIBulletinParser.
    """
    logger.info("Attempting to fetch real RBI Bulletin data...")

    try:
        req = urllib.request.Request(
            _RBI_BULLETIN_URL,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310 - trusted HTTPS endpoint
            charset = response.headers.get_content_charset() or "utf-8"
            base_html = response.read().decode(charset, errors="ignore")

        options = _extract_year_month_options(base_html)
        if not options:
            raise RuntimeError("Could not discover RBI Bulletin year/month archive options.")

        start = date_range.start
        end = date_range.end
        publications: list[dict[str, str]] = []

        current_html = base_html
        for year, month in options:
            if month == 0:
                period_start = date(year, 1, 1)
                period_end = date(year, 12, 31)
            else:
                period_start = date(year, month, 1)
                next_month = date(year + (month // 12), ((month % 12) + 1), 1) if month < 12 else date(year + 1, 1, 1)
                period_end = date.fromordinal(next_month.toordinal() - 1)

            if period_end < start or period_start > end:
                continue

            current_html = _post_year_month(current_html, year, month)
            publications.extend(_parse_bulletin_publications(current_html))

        dedup: dict[str, dict[str, str]] = {}
        for item in publications:
            obs_date = item["date"]
            if start <= date.fromisoformat(obs_date) <= end:
                dedup[obs_date] = item

        payload = {"publications": [dedup[k] for k in sorted(dedup.keys())]}
        logger.info(
            "Fetched %d RBI Bulletin publication event(s) for %s -> %s.",
            len(payload["publications"]),
            start,
            end,
        )
        return payload

    except Exception as e:
        logger.error(f"Failed to fetch or parse real RBI Bulletin data: {e}")
        raise RuntimeError(f"Real RBI Bulletin fetch failed: {e}") from e
