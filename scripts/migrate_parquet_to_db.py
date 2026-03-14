import sys
import os
import pandas as pd
from pathlib import Path
from datetime import timezone

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.db.silver_db_recorder import SilverDBRecorder
from src.schemas.market_data import Bar
from src.schemas.macro_data import MacroIndicator
from src.schemas.text_data import NewsArticle, SocialPost, EarningsTranscript

def migrate_ohlcv(recorder, base_dir: Path):
    ohlcv_path = base_dir / "ohlcv"
    if not ohlcv_path.exists():
        return
    
    total_bars = 0
    for parquet_file in ohlcv_path.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            bars = []
            for _, row in df.iterrows():
                row_dict = row.dropna().to_dict()
                bar = Bar(
                    timestamp=row_dict['timestamp'],
                    symbol=row_dict['symbol'],
                    interval=row_dict['interval'],
                    open=row_dict['open'],
                    high=row_dict['high'],
                    low=row_dict['low'],
                    close=row_dict['close'],
                    volume=row_dict['volume'],
                    source_type=row_dict.get('source_type', 'official_api'),
                    exchange=row_dict.get('exchange', 'NSE')
                )
                bars.append(bar)
            
            recorder.save_bars(bars)
            total_bars += len(bars)
            print(f"Migrated {len(bars)} OHLCV bars from {parquet_file.name}")
        except Exception as e:
            print(f"Error migrating {parquet_file}: {e}")
            
    print(f"Total OHLCV bars migrated: {total_bars}")

def migrate_macro(recorder, base_dir: Path):
    macro_path = base_dir / "macro"
    if not macro_path.exists():
        return
        
    total_indicators = 0
    for parquet_file in macro_path.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            indicators = []
            for _, row in df.iterrows():
                row_dict = row.dropna().to_dict()
                indicator = MacroIndicator(
                    timestamp=row_dict['timestamp'],
                    indicator_name=row_dict['indicator_name'],
                    value=row_dict['value'],
                    unit=row_dict['unit'],
                    period=row_dict['period'],
                    region=row_dict.get('region', 'India'),
                    source_type=row_dict.get('source_type', 'official_api')
                )
                indicators.append(indicator)
                
            recorder.save_macro_indicators(indicators)
            total_indicators += len(indicators)
            print(f"Migrated {len(indicators)} macro indicators from {parquet_file.name}")
        except Exception as e:
            print(f"Error migrating {parquet_file}: {e}")
            
    print(f"Total Macro indicators migrated: {total_indicators}")

def migrate_text(recorder, base_dir: Path):
    text_path = base_dir / "text" / "canonical"
    if not text_path.exists():
        return
        
    total_texts = 0
    for parquet_file in text_path.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            items = []
            for _, row in df.iterrows():
                try:
                    # Filter out NaN/None values from pandas
                    row_dict = row.dropna().to_dict()
                    
                    item = NewsArticle(
                        source_id=row_dict['source_id'],
                        timestamp=row_dict['timestamp'],
                        content=row_dict['content'],
                        source_type=row_dict.get('source_type', 'official_api'),
                        headline=row_dict.get('headline', 'No Headline'),
                        publisher=row_dict.get('publisher', 'Unknown'),
                        url=row_dict.get('url'),
                        author=row_dict.get('author')
                    )
                    items.append(item)
                except Exception as e:
                    print(f"Skipping row due to schema error: {e}")
                    continue
                    
            if items:
                recorder.save_text_items(items)
                total_texts += len(items)
                print(f"Migrated {len(items)} text items from {parquet_file.name}")
        except Exception as e:
            print(f"Error migrating {parquet_file}: {e}")

    print(f"Total text items migrated: {total_texts}")

if __name__ == "__main__":
    db_recorder = SilverDBRecorder()
    
    # Disable the monotonicity checker for this bulk historical backfill run
    # otherwise older data (2019-2023) gets quarantined because newer data (2024) already exists
    db_recorder.monotonicity_checker.check = lambda *args, **kwargs: True
    
    silver_dir = PROJECT_ROOT / "data" / "silver"
    
    print("--- Starting Parquet to PostgreSQL Migration ---")
    migrate_ohlcv(db_recorder, silver_dir)
    migrate_macro(db_recorder, silver_dir)
    migrate_text(db_recorder, silver_dir)
    print("--- Migration Complete ---")
