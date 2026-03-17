from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.connection import get_engine

logger = logging.getLogger(__name__)

MACRO_COLUMNS = [
    "CPI",
    "WPI",
    "IIP",
    "FII_FLOW",
    "DII_FLOW",
    "FX_RESERVES",
    "INDIA_US_10Y_SPREAD",
    "RBI_BULLETIN",
    "REPO_RATE",
    "US_10Y",
]


class RegimeDataLoader:
    """
    DB-first loader for regime modeling.
    Falls back to local Gold parquet files if DB is unavailable.
    """

    def __init__(self, database_url: str | None = None, gold_dir: str = "data/gold") -> None:
        self.database_url = database_url
        self.gold_dir = Path(gold_dir)

    def load_features(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        df = self._load_from_db(symbol=symbol, limit=limit)
        if df.empty:
            df = self._load_from_parquet(symbol=symbol, limit=limit)

        if df.empty:
            return df

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])

        return df.sort_values("timestamp").reset_index(drop=True)

    def build_labeled_dataset(self, symbol: str, limit: int = 2000) -> pd.DataFrame:
        """
        Day-1 helper: produce regime-labeled training rows from Gold features.
        Labels are deterministic heuristics until fully supervised labels are curated.
        """
        df = self.load_features(symbol=symbol, limit=limit)
        if df.empty:
            return df

        out = df.copy()
        if "close_log_return_zscore" not in out.columns:
            ret = pd.to_numeric(out.get("close_log_return"), errors="coerce")
            rolling = ret.rolling(window=30, min_periods=5)
            out["close_log_return_zscore"] = (ret - rolling.mean()) / rolling.std().replace(0, np.nan)

        if "rolling_vol_20" not in out.columns:
            ret = pd.to_numeric(out.get("close_log_return"), errors="coerce")
            out["rolling_vol_20"] = ret.rolling(window=20, min_periods=5).std()

        if "macro_directional_flag" not in out.columns:
            out["macro_directional_flag"] = 0.0

        out["rbi_zone"] = out.apply(self._map_rbi_zone, axis=1)
        out["regime_label"] = out.apply(self._label_row, axis=1)
        return out.reset_index(drop=True)

    @staticmethod
    def _map_rbi_zone(row: pd.Series) -> str:
        # If upstream enriches explicit RBI band distance, use it directly.
        band_distance = pd.to_numeric(row.get("rbi_band_distance"), errors="coerce")
        if pd.notna(band_distance):
            if abs(float(band_distance)) <= 0.25:
                return "inner_band"
            if abs(float(band_distance)) <= 0.75:
                return "mid_band"
            return "outer_band"

        macro_flag = pd.to_numeric(row.get("macro_directional_flag"), errors="coerce")
        if pd.isna(macro_flag) or float(macro_flag) == 0.0:
            return "inner_band"
        if abs(float(macro_flag)) <= 1.0:
            return "mid_band"
        return "outer_band"

    @staticmethod
    def _label_row(row: pd.Series) -> str:
        ret_z = pd.to_numeric(row.get("close_log_return_zscore"), errors="coerce")
        vol = pd.to_numeric(row.get("rolling_vol_20"), errors="coerce")
        rbi_zone = str(row.get("rbi_zone", "inner_band"))

        ret_z = 0.0 if pd.isna(ret_z) else float(ret_z)
        vol = 0.0 if pd.isna(vol) else float(vol)

        if abs(ret_z) >= 3.0 or vol >= 0.03:
            return "Crisis"
        if rbi_zone == "outer_band":
            return "RBI-Band transition"
        if ret_z >= 0.8:
            return "Bull"
        if ret_z <= -0.8:
            return "Bear"
        return "Sideways"

    def _load_from_db(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            engine = get_engine(self.database_url)
            query = """
                SELECT *
                FROM gold_features
                WHERE symbol = %(symbol)s
                ORDER BY timestamp DESC
                LIMIT %(limit)s
            """
            df = pd.read_sql(query, engine, params={"symbol": symbol, "limit": limit})
            if df.empty:
                return df
            return self._augment_macro_from_db(engine, df)
        except Exception as exc:
            logger.warning("RegimeDataLoader DB read failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def _augment_macro_from_db(self, engine, gold_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback-enrich Gold rows with asof macro joins when macro fields are sparse.
        """
        frame = gold_df.copy()
        if "timestamp" not in frame.columns:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            return frame

        # Detect sparse macro coverage in current Gold payload.
        existing_non_null = 0
        existing_total = 0
        for col in MACRO_COLUMNS:
            if col in frame.columns:
                existing_non_null += int(frame[col].notna().sum())
                existing_total += len(frame)
        coverage_ratio = (existing_non_null / existing_total) if existing_total > 0 else 0.0
        if coverage_ratio >= 0.20:
            return frame

        try:
            macro_query = """
                SELECT indicator_name, timestamp, value
                FROM macro_indicators
                ORDER BY timestamp ASC
            """
            macro_df = pd.read_sql(macro_query, engine)
        except Exception as exc:
            logger.warning("RegimeDataLoader macro enrichment query failed: %s", exc)
            return frame

        if macro_df.empty:
            return frame

        macro_df["timestamp"] = pd.to_datetime(macro_df["timestamp"], utc=True, errors="coerce")
        macro_df["value"] = pd.to_numeric(macro_df["value"], errors="coerce")
        macro_df = macro_df.dropna(subset=["timestamp", "value", "indicator_name"])
        if macro_df.empty:
            return frame

        pivot = (
            macro_df.pivot_table(
                index="timestamp",
                columns="indicator_name",
                values="value",
                aggfunc="last",
            )
            .sort_index()
            .reset_index()
        )
        if pivot.empty:
            return frame

        merged = pd.merge_asof(
            frame.sort_values("timestamp"),
            pivot.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            suffixes=("", "_macro"),
        )

        for col in MACRO_COLUMNS:
            source_col = f"{col}_macro" if f"{col}_macro" in merged.columns else col
            if source_col not in merged.columns:
                continue
            if col not in frame.columns:
                frame[col] = np.nan
            frame[col] = frame[col].where(frame[col].notna(), merged[source_col])

        return frame.sort_values("timestamp").reset_index(drop=True)

    def _load_from_parquet(self, symbol: str, limit: int) -> pd.DataFrame:
        if not self.gold_dir.exists():
            return pd.DataFrame()

        files = sorted(self.gold_dir.glob("**/*.parquet"))
        if not files:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for file_path in files[-30:]:
            try:
                frames.append(pd.read_parquet(file_path))
            except Exception as exc:
                logger.warning("Failed to read gold parquet %s: %s", file_path, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        if df.empty:
            return df

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        return df.tail(limit).reset_index(drop=True)
