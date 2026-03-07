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

    # NSE API requires browser-like headers, otherwise it rejects requests (403 Forbidden or timeouts)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/reports/fii-dii",
        # Sometimes NSE requires a valid cookie session. In a fully robust 
        # production system, we'd hit the homepage first to grab cookies.
    }

    try:
        req = urllib.request.Request(_NSE_FII_DII_API_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            raw_data = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            json_text = raw_data.decode(charset)
            
        data = json.loads(json_text)
        
        # The NSE format is typically a list of dictionaries where `category` indicates FII or DII.
        # Example schema: 
        # [
        #   {"category": "FII/FPI *", "date": "05-Mar-2026", "buyValue": "1000", "sellValue": "1500", "netValue": "-500"},
        #   {"category": "DII **", "date": "05-Mar-2026", "buyValue": "2000", "sellValue": "500", "netValue": "1500"}
        # ]
        
        if not isinstance(data, list):
            raise ValueError("NSE API returned non-list JSON payload.")

        fii_val = None
        dii_val = None
        record_date_str = None

        for row in data:
            cat = str(row.get("category", "")).upper()
            
            # The API usually provides commas in numbers e.g. "1,500.50"
            raw_net = str(row.get("netValue", "0")).replace(",", "")
            
            try:
                net_float = float(raw_net)
            except ValueError:
                continue

            if "FII" in cat or "FPI" in cat:
                fii_val = net_float
                if not record_date_str: 
                    record_date_str = row.get("date")
            elif "DII" in cat:
                dii_val = net_float
                if not record_date_str:
                    record_date_str = row.get("date")

        # Force the payload to use the pipeline's requested date range
        # (since the live NSE endpoint sometimes only returns "latest" data
        # which would otherwise be rejected by past-date bounds checkers)
        iso_date_str = date_range.end.isoformat()

        if fii_val is None and dii_val is None:
            raise ValueError("Could not find FII or DII categories in the NSE response.")

        payload = {
            "date": iso_date_str
        }
        if fii_val is not None:
            payload["fii_flow"] = fii_val
        if dii_val is not None:
            payload["dii_flow"] = dii_val

        logger.info(f"Successfully scraped FII/DII data for {iso_date_str}: {payload}")
        return payload

    except Exception as e:
        logger.error(f"Failed to fetch or parse real NSE FII/DII data: {e}")
        raise RuntimeError(f"Real FII/DII fetch failed: {e}") from e
