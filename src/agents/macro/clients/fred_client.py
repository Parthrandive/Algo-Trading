"""
Helpers for fetching public macro history from FRED CSV endpoints.

This module is intentionally lightweight and deterministic:
- Uses stable CSV endpoints (no API key required for reads).
- Returns observation-level rows with normalized keys.
- Never fabricates missing values.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.agents.macro.client import DateRange

logger = logging.getLogger(__name__)

_FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass(frozen=True)
class FredSeriesSpec:
    series_id: str
    frequency: str
    source_label: str
    unit_scale: float = 1.0


def _fetch_csv(series_id: str) -> str:
    url = f"{_FRED_CSV_BASE}?{urlencode({'id': series_id})}"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
            "Referer": "https://fred.stlouisfed.org/",
        },
    )
    with urlopen(req, timeout=30) as resp:  # nosec B310 - trusted HTTPS endpoint
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset)


def fetch_series_history(spec: FredSeriesSpec, date_range: DateRange) -> list[dict[str, Any]]:
    """
    Return normalized observations for a FRED series in [start, end] inclusive.

    Output row schema:
      {
        "date": "YYYY-MM-DD",
        "value": <float>,
        "release_date": None,
        "source": <source_label>,
        "frequency": <frequency>
      }
    """
    raw_csv = _fetch_csv(spec.series_id)
    reader = csv.DictReader(io.StringIO(raw_csv))

    if not reader.fieldnames or len(reader.fieldnames) < 2:
        raise RuntimeError(
            f"Unexpected CSV shape for FRED series {spec.series_id}: "
            f"columns={reader.fieldnames}"
        )

    date_col = reader.fieldnames[0]
    value_col = reader.fieldnames[1]

    start: date = date_range.start
    end: date = date_range.end

    rows: list[dict[str, Any]] = []
    for row in reader:
        raw_date = (row.get(date_col) or "").strip()
        raw_value = (row.get(value_col) or "").strip()

        if not raw_date or not raw_value or raw_value == ".":
            continue

        try:
            obs_date = date.fromisoformat(raw_date)
        except ValueError:
            continue

        if obs_date < start or obs_date > end:
            continue

        try:
            value = float(raw_value) * float(spec.unit_scale)
        except ValueError:
            continue

        rows.append(
            {
                "date": obs_date.isoformat(),
                "value": value,
                "release_date": None,
                "source": spec.source_label,
                "frequency": spec.frequency,
            }
        )

    logger.info(
        "Fetched %d row(s) from FRED series %s for %s -> %s",
        len(rows),
        spec.series_id,
        start,
        end,
    )
    return rows
