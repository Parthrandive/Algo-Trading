import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to expected NSE format (e.g., TCS -> TCS.NS)."""
    sym = symbol.strip().upper()
    if not sym:
        return sym
    is_plain = sym.replace("-", "").isalnum() and "." not in sym and "=" not in sym and "^" not in sym
    if is_plain:
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
