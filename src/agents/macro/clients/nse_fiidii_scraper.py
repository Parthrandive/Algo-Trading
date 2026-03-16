"""
Scraper for fetching real FII/DII daily flow data from the NSE website.
"""

import json
import logging
import urllib.request
from datetime import UTC, datetime
from typing import Any

from src.agents.macro.client import DateRange

logger = logging.getLogger(__name__)

# URL for fetching the FII/DII daily flows from NSE India
_NSE_FII_DII_API_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
_NSE_FII_DII_ALT_API_URL = "https://www.nseindia.com/api/fiidiiTradeNse"


def _parse_nse_date(date_str: str) -> datetime:
    # NSE payload format typically: 13-Mar-2026
    return datetime.strptime(date_str.strip(), "%d-%b-%Y").replace(tzinfo=UTC)


def _fetch_endpoint(url: str) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/reports/fii-dii",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:  # nosec B310 - trusted HTTPS endpoint
        raw_data = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        json_text = raw_data.decode(charset)
    payload = json.loads(json_text)
    if not isinstance(payload, list):
        raise ValueError(f"NSE API returned non-list payload from {url}")
    return payload

def fetch_real_fii_dii(date_range: DateRange) -> dict[str, Any]:
    """
    Fetches real FII and DII net values from NSE.

    Args:
        date_range: The date range to query (Note: The NSE live API typically just returns 
                    the latest trading day or a short rolling window, so we parse what it gives).

    Returns:
        dict: A payload parsable by FIIDIIParser, e.g.:
            {
                "date": "2026-03-05",
                "fii_flow": -1500.5,
                "dii_flow": 2000.0
            }

    Raises:
        RuntimeError: If the NSE API cannot be reached or returns an unexpected format.
    """
    logger.info("Attempting to fetch real FII/DII data from NSE API...")

    errors: list[str] = []
    for endpoint in (_NSE_FII_DII_API_URL, _NSE_FII_DII_ALT_API_URL):
        try:
            data = _fetch_endpoint(endpoint)

            fii_val = None
            dii_val = None
            record_date = None

            for row in data:
                cat = str(row.get("category", "")).upper()
                raw_net = str(row.get("netValue", "0")).replace(",", "")
                try:
                    net_float = float(raw_net)
                except ValueError:
                    continue

                if "FII" in cat or "FPI" in cat:
                    fii_val = net_float
                    if record_date is None and row.get("date"):
                        record_date = _parse_nse_date(str(row["date"])).date()
                elif "DII" in cat:
                    dii_val = net_float
                    if record_date is None and row.get("date"):
                        record_date = _parse_nse_date(str(row["date"])).date()

            if fii_val is None and dii_val is None:
                raise ValueError("Could not find FII/DII categories in response.")
            if record_date is None:
                raise ValueError("Could not parse record date from NSE payload.")

            if not (date_range.start <= record_date <= date_range.end):
                # NSE endpoint is latest-only; return an explicit out-of-range failure
                # rather than fabricating historical dates.
                raise ValueError(
                    f"NSE latest record date {record_date} is outside requested "
                    f"range [{date_range.start}, {date_range.end}]"
                )

            payload: dict[str, Any] = {"date": record_date.isoformat()}
            if fii_val is not None:
                payload["fii_flow"] = fii_val
            if dii_val is not None:
                payload["dii_flow"] = dii_val

            logger.info("Fetched NSE FII/DII payload from %s: %s", endpoint, payload)
            return payload
        except Exception as exc:  # pragma: no cover - network-dependent
            errors.append(f"{endpoint}: {exc}")
            continue

    raise RuntimeError("Real FII/DII fetch failed: " + " | ".join(errors))
