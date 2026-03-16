"""
MOSPI/OEA fetch helpers for CPI, WPI and IIP.

Notes:
- Official MOSPI/OEA pages are primarily document-based and do not expose a
  stable machine-readable historical API in this codebase.
- We therefore use official-page reachability checks plus a deterministic
  fallback to public FRED series for deep historical backfill.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.request import Request, urlopen

from src.agents.macro.client import DateRange
from src.agents.macro.clients.fred_client import FredSeriesSpec, fetch_series_history

logger = logging.getLogger(__name__)

_WPI_URL = "https://eaindustry.nic.in/"
_MOSPI_URL = "https://www.mospi.gov.in/"

# Public fallback series (FRED-hosted IMF/OECD data).
_FRED_MAP: dict[str, FredSeriesSpec] = {
    "CPI": FredSeriesSpec(
        series_id="INDCPIALLMINMEI",
        frequency="Monthly",
        source_label="fred:INDCPIALLMINMEI",
    ),
    "WPI": FredSeriesSpec(
        series_id="WPIATT01INM661N",
        frequency="Monthly",
        source_label="fred:WPIATT01INM661N",
    ),
    "IIP": FredSeriesSpec(
        series_id="INDPRINTO01IXOBM",
        frequency="Monthly",
        source_label="fred:INDPRINTO01IXOBM",
    ),
}


def _probe_official_endpoint(indicator: str) -> None:
    """Best-effort official endpoint probe; raises on reachability failures."""
    url = _WPI_URL if indicator == "WPI" else _MOSPI_URL
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=20):  # nosec B310 - trusted HTTPS endpoint
        return


def fetch_real_mospi_data(indicator: str, date_range: DateRange) -> dict[str, Any]:
    """
    Fetch normalized historical rows for CPI/WPI/IIP.

    Returns parser-compatible payload:
      {"data": [{"date": "...", "value": ...}, ...]}
    """
    indicator = indicator.upper().strip()
    if indicator not in _FRED_MAP:
        raise ValueError(f"Unsupported MOSPI indicator: {indicator!r}")

    # Keep official-source reachability check in place.
    _probe_official_endpoint(indicator)

    fallback_rows = fetch_series_history(_FRED_MAP[indicator], date_range)
    return {
        "data": [
            {
                "date": row["date"],
                "value": row["value"],
                "release_date": row.get("release_date"),
                "source": row.get("source"),
                "frequency": row.get("frequency"),
            }
            for row in fallback_rows
        ]
    }
