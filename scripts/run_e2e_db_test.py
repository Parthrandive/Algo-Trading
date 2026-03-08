import os
import sys
import argparse
from sqlalchemy import text
from datetime import datetime
from src.db.connection import get_engine, get_session

def print_header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def run_e2e_test():
    engine = get_engine()
    
    # 1. Clear DB to prove it's a fresh run
    print_header("1. Wiping DB Clean for E2E Test")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE ohlcv_bars, ticks, macro_indicators, text_items, gold_features RESTART IDENTITY CASCADE;"))
        print("Emptied all Silver and Gold tables.")

    # 2. Run Bronze -> Silver (All Agents)
    print_header("2. Running Unified Silver Ingestion (Market, Macro, Textual)")
    
    print("\n--- Market Data (Backfill) ---")
    old_argv = sys.argv
    sys.argv = [
        "backfill_historical.py",
        "--symbols", "RELIANCE,INFY",
        "--days", "3",
        "--interval", "1h",
        "--workers", "2",
        "--skip-recent-hours", "1",
        "--force-refresh"
    ]
    from scripts.backfill_historical import run as run_backfill
    exit_code = run_backfill(sys.argv[1:])
    if exit_code != 0:
        print(f"Market data ingestion failed with code {exit_code}")
    else:
        print("Market data ingestion succeeded.")
        
    print("\n--- Macro Data Agent ---")
    from src.agents.macro.run_real_pipeline import run_macro
    try:
        run_macro()
        print("Macro data ingestion succeeded.")
    except Exception as e:
        print(f"Macro data ingestion failed: {e}")

    print("\n--- Textual Data Agent ---")
    from src.agents.textual.textual_data_agent import TextualDataAgent
    from src.db.silver_db_recorder import SilverDBRecorder
    try:
        text_agent = TextualDataAgent.from_default_components(recorder=SilverDBRecorder())
        text_agent.run_once()
        print("Textual data ingestion succeeded.")
    except Exception as e:
        print(f"Textual data ingestion failed: {e}")

    # 3. Run Silver -> Gold (Preprocessing Agent)
    print_header("3. Running Preprocessing Pipeline (Silver -> Gold)")
    from src.agents.preprocessing.pipeline import PreprocessingPipeline
    try:
        pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
        snapshot_id = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output = pipeline.process_snapshot(
            market_source_path="db_virtual",
            macro_source_path="db_virtual",
            text_source_path="db_virtual",
            snapshot_id=snapshot_id
        )
        print(f"Preprocessing pipeline completed. Output Rows: {len(output.feature_df)}")
    except Exception as e:
        print(f"Preprocessing Pipeline failed: {e}")

    # 4. Verify DB Results
    print_header("4. Database Verification Results")
    Session = get_session(engine)
    with Session() as session:
        market_count = session.execute(text("SELECT COUNT(*) FROM ohlcv_bars")).scalar()
        macro_count = session.execute(text("SELECT COUNT(*) FROM macro_indicators")).scalar()
        text_count = session.execute(text("SELECT COUNT(*) FROM text_items")).scalar()
        gold_count = session.execute(text("SELECT COUNT(*) FROM gold_features")).scalar()
        
        print(f"Silver Layer - Market Bars: {market_count}")
        print(f"Silver Layer - Macro Indicators: {macro_count}")
        print(f"Silver Layer - Text Items: {text_count}")
        print(f"Gold Layer - Processed Features: {gold_count}")
        
        if market_count > 0 and macro_count > 0 and text_count > 0 and gold_count > 0:
            print("\n✅ SUCCESS: End-to-End DB Pipeline is fully functional!")
        else:
            print("\n❌ FAILURE: One or more data layers are missing records.")

if __name__ == "__main__":
    run_e2e_test()
