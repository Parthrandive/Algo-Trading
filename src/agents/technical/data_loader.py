import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.agents.technical.nsemine_fetcher import NseMineFetcher

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

MACRO_RELEASE_DELAYS = {
    "CPI": timedelta(days=14),
    "WPI": timedelta(days=14),
    "IIP": timedelta(days=42),
    "FII_FLOW": timedelta(hours=4),
    "DII_FLOW": timedelta(hours=4),
    "FX_RESERVES": timedelta(days=7),
    "INDIA_US_10Y_SPREAD": timedelta(hours=6),
    "RBI_BULLETIN": timedelta(hours=24),
    "REPO_RATE": timedelta(days=1),
    "US_10Y": timedelta(hours=6),
}


class DataLoader:
    """
    Loads historical Silver/Gold tier OHLCV bars for the Technical Agent.
    """

    def __init__(self, db_url: str, macro_coverage_threshold: Optional[float] = None):
        """
        Initialize the DataLoader with a database connection.

        Args:
            db_url (str): SQLAlchemy database connection string.
            macro_coverage_threshold (float, optional): minimum macro coverage ratio
                (0-1) required for a macro feature to be train-ready.
        """
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.macro_coverage_threshold = (
            float(macro_coverage_threshold)
            if macro_coverage_threshold is not None
            else float(os.getenv("MACRO_FEATURE_COVERAGE_THRESHOLD", "0.60"))
        )
        self.last_macro_quality_report: dict[str, dict] = {}
        self.last_macro_excluded_features: list[str] = []

    def load_from_nse(self, symbol: str, from_date: str, to_date: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch historical blocks purely from NSE via nsemine."""
        return NseMineFetcher.fetch_historical(symbol, from_date, to_date, interval)

    def _is_postgres_engine(self) -> bool:
        url = str(self.engine.url).lower()
        return url.startswith("postgresql")

    @staticmethod
    def _drop_all_null_vwap(df: pd.DataFrame) -> pd.DataFrame:
        if "vwap" in df.columns and df["vwap"].isna().all():
            return df.drop(columns=["vwap"])
        return df

    def _normalize_macro_frame(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        normalized = macro_df.copy()
        normalized["indicator_name"] = normalized["indicator_name"].astype(str).str.strip().str.upper()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
        normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
        normalized = normalized.dropna(subset=["indicator_name", "timestamp", "value"]).copy()
        if normalized.empty:
            return normalized

        if "release_date" in normalized.columns:
            normalized["release_date"] = pd.to_datetime(
                normalized["release_date"], utc=True, errors="coerce"
            )
        else:
            normalized["release_date"] = pd.NaT

        normalized["series_name"] = normalized["indicator_name"]
        normalized["observation_date"] = normalized["timestamp"].dt.date
        normalized["source"] = normalized.get("source", normalized.get("source_type", "unknown"))
        normalized["frequency"] = normalized.get("frequency", normalized.get("period", "Unknown"))
        normalized["release_ts"] = pd.to_datetime(
            normalized["release_date"], utc=True, errors="coerce"
        )

        missing_release = normalized["release_ts"].isna()
        if missing_release.any():
            for series_name, delay in MACRO_RELEASE_DELAYS.items():
                mask = missing_release & (normalized["series_name"] == series_name)
                if mask.any():
                    normalized.loc[mask, "release_ts"] = normalized.loc[mask, "timestamp"] + delay
            unresolved = normalized["release_ts"].isna()
            if unresolved.any():
                normalized.loc[unresolved, "release_ts"] = normalized.loc[unresolved, "timestamp"]

        normalized["release_ts"] = pd.to_datetime(normalized["release_ts"], utc=True, errors="coerce")

        normalized = (
            normalized.sort_values(["series_name", "release_ts", "timestamp"])
            .drop_duplicates(subset=["series_name", "release_ts"], keep="last")
            .reset_index(drop=True)
        )
        return normalized

    def _build_macro_quality_report(
        self,
        joined_df: pd.DataFrame,
        normalized_macro: pd.DataFrame,
    ) -> dict[str, dict]:
        total_rows = len(joined_df)
        threshold_pct = self.macro_coverage_threshold * 100.0
        report: dict[str, dict] = {}

        for feature in MACRO_COLUMNS:
            subset = normalized_macro[normalized_macro["series_name"] == feature]
            first_date = None if subset.empty else str(subset["observation_date"].min())
            last_date = None if subset.empty else str(subset["observation_date"].max())

            non_null = int(joined_df[feature].notna().sum()) if feature in joined_df.columns else 0
            missing = int(total_rows - non_null)
            coverage_pct = (100.0 * non_null / total_rows) if total_rows else 0.0
            train_ready = bool(non_null > 0 and coverage_pct >= threshold_pct)

            report[feature] = {
                "first_available_date": first_date,
                "last_available_date": last_date,
                "row_count": int(len(subset)),
                "coverage_pct": round(float(coverage_pct), 4),
                "missing_after_join": missing,
                "train_ready": train_ready,
            }

        return report

    def _augment_with_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Release-aware as-of join of macro indicators onto market bars.

        Rules:
        - Never leak future macro releases.
        - Only forward-fill after release timestamp/date.
        - Never backfill before first real observation.
        - Apply feature gating based on configurable coverage threshold.
        """
        self.last_macro_quality_report = {}
        self.last_macro_excluded_features = []

        if df.empty or not self._is_postgres_engine():
            return self._drop_all_null_vwap(df)
        if "timestamp" not in df.columns:
            return self._drop_all_null_vwap(df)

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if out.empty:
            return self._drop_all_null_vwap(out)

        min_ts = out["timestamp"].min()
        max_ts = out["timestamp"].max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            return self._drop_all_null_vwap(out)

        try:
            macro_query = text(
                """
                SELECT *
                FROM macro_indicators
                WHERE timestamp <= :max_ts
                ORDER BY timestamp ASC
                """
            )
            macro_df = pd.read_sql(
                macro_query,
                self.engine,
                params={"max_ts": max_ts.to_pydatetime()},
            )
        except Exception as exc:
            logger.warning("Macro augmentation query failed: %s", exc)
            return self._drop_all_null_vwap(out)

        if macro_df.empty:
            return self._drop_all_null_vwap(out)

        macro_df = self._normalize_macro_frame(macro_df)
        if macro_df.empty:
            return self._drop_all_null_vwap(out)

        # Per-series release-aware as-of join.
        for col in MACRO_COLUMNS:
            series = (
                macro_df.loc[macro_df["series_name"] == col, ["release_ts", "value"]]
                .rename(columns={"release_ts": "effective_timestamp"})
                .dropna(subset=["effective_timestamp", "value"])
                .sort_values("effective_timestamp")
            )
            if series.empty:
                continue

            merged = pd.merge_asof(
                out.sort_values("timestamp"),
                series,
                left_on="timestamp",
                right_on="effective_timestamp",
                direction="backward",
            )
            out[col] = merged["value"]

        report = self._build_macro_quality_report(out, macro_df)
        self.last_macro_quality_report = report
        excluded = sorted([name for name, stats in report.items() if not stats["train_ready"]])
        self.last_macro_excluded_features = excluded
        out.attrs["macro_quality_report"] = report
        out.attrs["macro_excluded_features"] = excluded

        if excluded:
            logger.info(
                "Macro feature gating excluded %d feature(s) below %.1f%% coverage: %s",
                len(excluded),
                self.macro_coverage_threshold * 100.0,
                excluded,
            )
            for col in excluded:
                if col in out.columns:
                    out[col] = pd.NA

        out = out.sort_values("timestamp").reset_index(drop=True)
        return self._drop_all_null_vwap(out)

    def load_historical_bars(
        self,
        symbol: str,
        limit: Optional[int] = None,
        use_nse_fallback: bool = False,
        min_fallback_rows: int = 100,
        interval: str = "1d",
        include_macro: bool = True,
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data for a specific symbol from the sentinel_db.ohlcv_bars table.
        If use_nse_fallback is True and the DB returns fewer than min_fallback_rows,
        it will attempt to fetch the last 1 year of data from the NSE natively.

        Args:
            symbol (str): The stock symbol (e.g., 'TATASTEEL.NS')
            limit (int, optional): Maximum number of rows to retrieve.
            use_nse_fallback (bool, optional): Allow hitting NSE if local data is sparse.
            min_fallback_rows (int, optional): Threshold for triggering NSE fallback.
            interval (str, optional): The candle interval, like '1d', '1h'. Defaults to '1d'.

        Returns:
            pd.DataFrame: DataFrame containing OHLCV features.
        """
        if not symbol:
            raise ValueError("`symbol` must be a non-empty string.")

        query = """
            SELECT *
            FROM ohlcv_bars
            WHERE symbol = :symbol AND interval = :interval
            ORDER BY timestamp ASC
        """
        params = {"symbol": symbol, "interval": interval}

        if limit is not None:
            if limit <= 0:
                raise ValueError("`limit` must be a positive integer when provided.")
            query += " LIMIT :limit"
            params["limit"] = int(limit)

        try:
            df = pd.read_sql(text(query), self.engine, params=params)
            if include_macro:
                df = self._augment_with_macro(df)
            else:
                df = self._drop_all_null_vwap(df)

            # Check for fallback
            if use_nse_fallback and len(df) < min_fallback_rows:
                logger.info(f"DB returned {len(df)} rows, less than {min_fallback_rows}. Trying NSE fallback...")
                import datetime as dt
                to_dt = dt.datetime.now()
                from_dt = to_dt - dt.timedelta(days=365)
                nse_df = self.load_from_nse(
                    symbol=symbol,
                    from_date=from_dt.strftime("%d-%m-%Y"),
                    to_date=to_dt.strftime("%d-%m-%Y"),
                    interval=interval,
                )
                if not nse_df.empty:
                    if include_macro:
                        nse_df = self._augment_with_macro(nse_df)
                    else:
                        nse_df = self._drop_all_null_vwap(nse_df)
                    if limit is not None:
                        return nse_df.tail(limit).copy()
                    return nse_df

            return df
        except Exception as e:
            logger.error(f"DB Load failed: {e}. Trying NSE if permitted.")
            if use_nse_fallback:
                import datetime as dt
                to_dt = dt.datetime.now()
                from_dt = to_dt - dt.timedelta(days=365)
                nse_df = self.load_from_nse(
                    symbol=symbol,
                    from_date=from_dt.strftime("%d-%m-%Y"),
                    to_date=to_dt.strftime("%d-%m-%Y"),
                    interval=interval,
                )
                if include_macro:
                    nse_df = self._augment_with_macro(nse_df)
                else:
                    nse_df = self._drop_all_null_vwap(nse_df)
                return nse_df
            raise RuntimeError(f"Failed to load historical bars: {str(e)}")
