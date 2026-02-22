import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.db.connection import get_engine
from sqlalchemy import text

def test_read():
    engine = get_engine()
    print(f"Connecting to: {engine.url}")
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT count(*) FROM ohlcv_bars;"))
        count = result.scalar()
        print(f"Number of rows in ohlcv_bars: {count}")

if __name__ == "__main__":
    test_read()
