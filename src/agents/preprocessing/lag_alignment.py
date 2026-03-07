import hashlib
import json
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.schemas.preprocessing_data import TransformOutput

class LagAligner:
    """
    Safely aligns irregular/slow macro data onto fast market data.
    Enforces publication lag to prevent look-ahead bias (Section 5.3 Leakage tests).
    """
    def __init__(self, publication_delays: Optional[Dict[str, timedelta]] = None):
        # Default publication lags for macro indicators (e.g. CPI takes ~14 days to publish after observation)
        self.publication_delays = publication_delays or {
            "CPI": timedelta(days=14),
            "WPI": timedelta(days=14),
            "IIP": timedelta(days=42),
            "FII_FLOW": timedelta(hours=4),
            "DII_FLOW": timedelta(hours=4),
            "FX_RESERVES": timedelta(days=7),
            "INDIA_US_10Y_SPREAD": timedelta(hours=6),
            "RBI_BULLETIN": timedelta(hours=24)
        }

    def align(self, market_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges macro data to market data using Pandas asof merge.
        Crucial: It ensures that market rows only see macro data that was PUBLISHED BEFORE the market timestamp.
        """
        if market_df.empty:
            return market_df
            
        if macro_df.empty:
            return market_df

        # Work on copies and force nanosecond precision so merge_asof keys match
        market_df = market_df.copy()
        macro_df = macro_df.copy()
        market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        macro_df["timestamp"] = pd.to_datetime(macro_df["timestamp"], utc=True).astype("datetime64[ns, UTC]")

        # 1. Pivot Macro dataframe so each indicator is a column
        pivot_macro = macro_df.pivot_table(
            index="timestamp", 
            columns="indicator_name", 
            values="value", 
            aggfunc="last"
        ).reset_index()
        pivot_macro["timestamp"] = pd.to_datetime(pivot_macro["timestamp"], utc=True).astype("datetime64[ns, UTC]")

        # 2. Shift the timestamps forward by their respective publication delays.
        # This creates 'effective_time' meaning "when did this data become known to the market?"
        aligned_dfs = []
        for col in pivot_macro.columns:
            if col == "timestamp":
                continue
                
            delay = self.publication_delays.get(col, timedelta(0))
            
            # Create a localized dataframe for the indicator shifted forward
            ind_df = pivot_macro[["timestamp", col]].dropna().copy()
            ind_df["effective_time"] = (ind_df["timestamp"] + delay).astype("datetime64[ns, UTC]")
            ind_df = ind_df.sort_values("effective_time")
            
            # Merge this indicator into the market frame using standard backward asof
            # Match effective_time <= market timestamp
            market_df = market_df.sort_values("timestamp")
            market_df["timestamp"] = market_df["timestamp"].astype("datetime64[ns, UTC]")
            market_df = pd.merge_asof(
                market_df, 
                ind_df[["effective_time", col]], 
                left_on="timestamp", 
                right_on="effective_time", 
                direction="backward"
            )
            # Cleanup intermediate timing column 
            market_df = market_df.drop(columns=["effective_time"], errors="ignore")

        if "symbol" in market_df.columns:
            market_df = market_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        return market_df

class CorporateActionValidator:
    """
    Adjusts historical series for corporate actions and validates against the schema parameters.
    Section 5.3 requirement: Corporate action adjustments validated.
    """
    def __init__(self):
        pass

    def apply_adjustments(self, market_df: pd.DataFrame, corporate_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Very crude back-adjustment example for splits/bonuses for Phase 1.
        Adjusts historical prices iteratively moving backward from the action date.
        """
        if market_df.empty or corporate_actions_df.empty:
            return market_df
            
        market_df = market_df.copy()
        market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)
        market_df = market_df.sort_values(by=["symbol", "timestamp"])
        
        ca_df = corporate_actions_df.copy()
        ca_df["ex_date"] = pd.to_datetime(ca_df["ex_date"], utc=True)
        
        # Supported types: SPLIT, BONUS  (Dividends ignored for simple price ratios in this phase)
        for _, row in ca_df[ca_df["action_type"].isin(["SPLIT", "BONUS"])].iterrows():
            sym = row["symbol"]
            ex_date = row["ex_date"]
            ratio = row["ratio"]
            
            if pd.isna(ratio) or ratio <= 0:
                continue

            # Identify all rows for the symbol before the ex_date
            mask = (market_df["symbol"] == sym) & (market_df["timestamp"] < ex_date)
            
            # Adjust price thresholds backwards
            for col in ["open", "high", "low", "close"]:
                if col in market_df.columns:
                    market_df.loc[mask, col] = market_df.loc[mask, col] / ratio
            
            # Adjust volume equivalently up (money is same, shares are more)
            if "volume" in market_df.columns:
                 market_df.loc[mask, "volume"] = market_df.loc[mask, "volume"] * ratio
                 
        return market_df
