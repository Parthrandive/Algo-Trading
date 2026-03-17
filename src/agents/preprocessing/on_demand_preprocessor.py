import os
import logging
from datetime import datetime, timezone
import pandas as pd
from typing import Optional

from src.db.connection import get_engine
from src.agents.preprocessing.pipeline import PreprocessingPipeline
from scripts.backfill_historical import run as run_backfill

logger = logging.getLogger(__name__)

class OnDemandPreprocessor:
    """
    Provides an API for Phase 2 agents to request cleanly processed Gold
    features dynamically during live market hours.
    Allows for TTL freshness caching and auto-backfilling missing symbols.
    """
    def __init__(self, config_path: str = "configs/transform_config_v1.json"):
        self.engine = get_engine()
        self.config_path = config_path
        self.ttl_seconds = int(os.environ.get("GOLD_FRESHNESS_TTL_SECONDS", "300"))

    def get_clean_features(self, symbol: str) -> pd.DataFrame:
        """
        Retrieves clean Gold features for a specific symbol on-demand.
        If the data is stale or missing, it triggers an auto-fetch and preprocessing.
        """
        logger.info(f"On-demand request received for {symbol}")
        
        # 1. Check freshness
        if self._is_data_fresh(symbol):
            logger.info(f"Returning cached Gold data for {symbol} (Age < {self.ttl_seconds}s)")
            return self._fetch_gold_data(symbol)

        logger.info(f"Gold data for {symbol} is stale or missing. Initiating on-demand processing.")
        
        # 2. Check if we need to auto-backfill Silver Data (e.g. new symbol never processed)
        silver_count = pd.read_sql(f"SELECT COUNT(*) FROM ohlcv_bars WHERE symbol='{symbol}'", self.engine).iloc[0, 0]
        if silver_count == 0:
            logger.info(f"No Silver data found for {symbol}. Triggering auto-backfill.")
            self._trigger_backfill(symbol)

        try:
            from src.db.queries import get_market_data_quality

            quality = get_market_data_quality(symbol, "1h", dataset_type="historical")
            if quality is not None and not quality.get("train_ready"):
                raise ValueError(
                    f"Symbol {symbol} is not train-ready: {quality.get('details_json')}"
                )
        except Exception as exc:
            logger.error("Historical quality gate rejected %s: %s", symbol, exc)
            raise
            
        # 3. Clean and Save to Gold
        pipeline = PreprocessingPipeline(config_path=self.config_path)
        snapshot_id = f"ondemand_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        try:
            output = pipeline.process_snapshot(
                market_source_path="db_virtual",
                macro_source_path="db_virtual",
                text_source_path="db_virtual",
                snapshot_id=snapshot_id,
                symbol_filter=symbol
            )
            result_df = pd.DataFrame(output.records)
            logger.info(f"Successfully processed {len(result_df)} rows for {symbol} on-demand.")
            return result_df
        except Exception as e:
            logger.error(f"On-demand preprocessing failed for {symbol}: {e}")
            raise

    def _is_data_fresh(self, symbol: str) -> bool:
        """Determines if the gold data for this symbol was recently updated via on-demand requests."""
        try:
            from sqlalchemy import text
            # We look for the most recent ondemand snapshot for this symbol
            query = text(f"SELECT snapshot_id FROM gold_features WHERE symbol='{symbol}' AND snapshot_id LIKE 'ondemand_%' ORDER BY timestamp DESC LIMIT 1")
            df = pd.read_sql(query, self.engine)
            if df.empty:
                return False
                
            snapshot_id = df.iloc[0]['snapshot_id']
            # parse UTC timestamp from 'ondemand_20260308_020428'
            time_str = snapshot_id.split('_', 1)[1]
            last_run = datetime.strptime(time_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            
            age_seconds = (datetime.now(timezone.utc) - last_run).total_seconds()
            return age_seconds <= self.ttl_seconds
            
        except Exception as e:
            logger.warning(f"Freshness check failed: {e}. Defaulting to stale.")
            return False

    def _fetch_gold_data(self, symbol: str) -> pd.DataFrame:
        query = f"SELECT * FROM gold_features WHERE symbol='{symbol}' ORDER BY timestamp ASC"
        return pd.read_sql(query, self.engine)

    def _trigger_backfill(self, symbol: str) -> None:
        """Runs the historical backfill specifically for this symbol."""
        args = [
            "--symbols", symbol,
            "--days", "14",
            "--interval", "1h",
            "--workers", "1",
            "--skip-recent-hours", "1", 
            "--force-refresh"
        ]
        
        import sys
        old_argv = sys.argv
        sys.argv = ["backfill_historical.py"] + args
        try:
            exit_code = run_backfill(sys.argv[1:])
            if exit_code != 0:
                 logger.error(f"Auto-backfill failed for {symbol} with exit code {exit_code}")
        finally:
            sys.argv = old_argv
