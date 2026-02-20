import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

CURRENCY_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "CNY",
    "EUR",
    "GBP",
    "HKD",
    "INR",
    "JPY",
    "NZD",
    "SGD",
    "USD",
}


def _looks_like_forex_pair(sym: str) -> bool:
    if len(sym) != 6 or not sym.isalpha():
        return False
    base = sym[:3]
    quote = sym[3:]
    return base in CURRENCY_CODES and quote in CURRENCY_CODES


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbols for this project:
    - NSE equities: TCS -> TCS.NS
    - FX pairs: USDINR -> USDINR=X
    """
    sym = symbol.strip().upper()
    if not sym:
        return sym
    if "." in sym or "=" in sym or "^" in sym:
        return sym
    if _looks_like_forex_pair(sym):
        return f"{sym}=X"
    if sym.replace("-", "").isalnum():
        return f"{sym}.NS"
    return sym

def get_latest_local_timestamp(symbol: str, base_dir: str = "data/silver/ohlcv") -> Optional[datetime]:
    """
    Find the maximum timestamp for a symbol currently stored in the silver layer.
    Useful for gap detection before fetching new data.
    """
    normalized = normalize_symbol(symbol)
    
    # Check exact normalized first
    symbol_dir = Path(base_dir) / normalized
    if not symbol_dir.exists():
        # Try exact input alias
        symbol_dir = Path(base_dir) / symbol
        if not symbol_dir.exists():
            return None
            
    # Find all parquet partitions
    files = sorted(symbol_dir.rglob("*.parquet"))
    if not files:
        return None
        
    # The last file by naming convention (YYYY-MM-DD.parquet) should hold the latest data.
    latest_file = files[-1]
    try:
        df = pd.read_parquet(latest_file)
        if "timestamp" in df.columns:
            max_ts = df["timestamp"].max()
            if not pd.isna(max_ts):
                ts = max_ts.to_pydatetime()
                # Ensure it carries timezone info if it was saved with it
                return ts
    except Exception as e:
        logger.error(f"Failed to read parquet for gap detection {latest_file}: {e}")
        
    return None
