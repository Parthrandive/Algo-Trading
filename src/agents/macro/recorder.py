"""
MacroSilverRecorder — Parquet + optional DB persistence for MacroIndicator records.

Follows the same architectural pattern as SentinelAgent's SilverRecorder
(data/silver/ohlcv/...) but adapted for macro indicators:

Parquet partition scheme:
    data/silver/macro/<indicator_name>/<YYYY>/<MM>/<YYYY-MM-DD>.parquet

Deduplication key: (indicator_name, timestamp)

Optional DB persistence via ``SilverDBRecorder.save_macro_indicators()``.
Pass ``db_recorder=None`` (default) to skip DB writes (e.g. in unit tests).

Provenance guarantee:
    Every MacroIndicator written here already carries full provenance from
    the upstream client (source_type, schema_version, quality_status,
    ingestion_timestamp_utc/ist). This recorder does NOT modify those fields.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Protocol

import pandas as pd

from src.schemas.macro_data import MacroIndicator

logger = logging.getLogger(__name__)


class MacroDBRecorderProtocol(Protocol):
    """Minimal protocol for the optional DB backend."""

    def save_macro_indicators(self, indicators: List[MacroIndicator]) -> None: ...


class MacroSilverRecorder:
    """
    Silver Layer recorder for MacroIndicator records.

    Persists validated ``MacroIndicator`` objects to Parquet files partitioned
    by indicator → year → month, and optionally to a database.

    Parameters
    ----------
    base_dir:
        Root directory for silver-layer Parquet files.
        Defaults to ``data/silver/macro``.
    quarantine_dir:
        Directory for records that fail dedup / validation.
        Defaults to ``data/quarantine/macro``.
    db_recorder:
        Optional object satisfying ``MacroDBRecorderProtocol``.
        If provided, ``save_macro_indicators`` is called after Parquet write.
    """

    def __init__(
        self,
        base_dir: str = "data/silver/macro",
        quarantine_dir: str = "data/quarantine/macro",
        db_recorder: Optional[MacroDBRecorderProtocol] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.db_recorder = db_recorder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_indicators(self, indicators: List[MacroIndicator]) -> None:
        """
        Persist a batch of ``MacroIndicator`` records.

        Steps:
        1. Validate schema_version == "1.1" — quarantine non-conforming rows.
        2. Write to Parquet (dedup on indicator_name + timestamp).
        3. If db_recorder is set, persist to DB.

        Parameters
        ----------
        indicators:
            Records to persist. May be empty (no-op).
        """
        if not indicators:
            return

        valid: List[MacroIndicator] = []
        quarantined: List[MacroIndicator] = []

        for ind in indicators:
            if ind.schema_version != "1.1":
                logger.warning(
                    "Quarantining record with unexpected schema_version=%r "
                    "(indicator=%s, timestamp=%s)",
                    ind.schema_version,
                    ind.indicator_name.value,
                    ind.timestamp,
                )
                quarantined.append(ind)
            else:
                valid.append(ind)

        if quarantined:
            self._write_parquet(quarantined, self.quarantine_dir)

        if valid:
            self._write_parquet(valid, self.base_dir)
            if self.db_recorder is not None:
                try:
                    self.db_recorder.save_macro_indicators(valid)
                except Exception as exc:
                    logger.error(
                        "DB persist failed for %d macro indicator(s): %s",
                        len(valid),
                        exc,
                    )
                    raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_parquet(
        self, indicators: List[MacroIndicator], base_output_dir: Path
    ) -> None:
        """Convert indicators to DataFrame and write partitioned Parquet."""
        data = [ind.model_dump(mode="json") for ind in indicators]
        df = pd.DataFrame(data)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")

        for indicator_name, indicator_group in df.groupby("indicator_name"):
            for date_str, day_group in indicator_group.groupby("date_str"):
                ts = pd.to_datetime(date_str)
                year = str(ts.year)
                month = f"{ts.month:02d}"

                target_dir = base_output_dir / str(indicator_name) / year / month
                target_dir.mkdir(parents=True, exist_ok=True)

                file_path = target_dir / f"{date_str}.parquet"

                day_group = day_group.drop(columns=["date_str"])

                if file_path.exists():
                    try:
                        existing_df = pd.read_parquet(file_path)
                        combined_df = pd.concat([existing_df, day_group])
                        # Dedup: keep latest ingested record for same (indicator, timestamp)
                        combined_df = combined_df.drop_duplicates(
                            subset=["indicator_name", "timestamp"], keep="last"
                        )
                    except Exception as exc:
                        logger.error(
                            "Error reading existing parquet %s: %s — overwriting",
                            file_path,
                            exc,
                        )
                        combined_df = day_group
                else:
                    combined_df = day_group

                combined_df.to_parquet(file_path, index=False)
                logger.debug(
                    "Wrote %d record(s) → %s", len(combined_df), file_path
                )
