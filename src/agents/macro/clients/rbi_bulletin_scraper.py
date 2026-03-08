"""
Scraper for fetching real RBI Bulletin publications.
"""

import logging
import urllib.request
from typing import Any
import pandas as pd

from src.agents.macro.client import DateRange

logger = logging.getLogger(__name__)

_RBI_BULLETIN_URL = "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx"

def fetch_real_rbi_bulletin(date_range: DateRange) -> dict[str, Any]:
    """
    Fetches real RBI Bulletin titles from the RBI website.
    
    Args:
        date_range: The date range to query.
    Returns:
        dict: A payload parsable by RBIBulletinParser.
    """
    logger.info("Attempting to fetch real RBI Bulletin data...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        req = urllib.request.Request(_RBI_BULLETIN_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            raw_data = response.read()

        dfs = pd.read_html(raw_data)
        
        iso_date_str = date_range.end.isoformat()
        
        title = "RBI Bulletin (Real Web Scrape)"
        if len(dfs) > 1 and not dfs[1].empty:
             title = f"RBI Bulletin: {dfs[1].iloc[0, 0]}"
            
        payload = {
            "publications": [
                {
                    "date": iso_date_str,
                    "title": str(title)[:100],
                }
            ]
        }
        
        logger.info(f"Successfully scraped RBI Bulletin data for {iso_date_str}: {payload}")
        return payload

    except Exception as e:
        logger.error(f"Failed to fetch or parse real RBI Bulletin data: {e}")
        raise RuntimeError(f"Real RBI Bulletin fetch failed: {e}") from e
