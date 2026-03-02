from typing import List, Dict, Any, Optional
import hashlib
import json
import logging

import pandas as pd

from src.agents.preprocessing.loader import MacroLoader, MarketLoader
from src.agents.preprocessing.lag_alignment import LagAligner, CorporateActionValidator
from src.agents.preprocessing.normalizers import LogReturnNormalizer, ZScoreNormalizer, MinMaxNormalizer, DirectionalChangeDetector
from src.agents.preprocessing.transform_graph import TransformGraph
from src.schemas.preprocessing_data import TransformOutput
from src.agents.preprocessing.reproducibility import ReproducibilityHasher
from src.utils.latency import timed

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Orchestrator for the Preprocessing Agent. 
    1. Loads Silver Data
    2. Adjusts Corporate Actions
    3. Aligns Macro Indicators (with publication lag)
    4. Applies Transform DAG
    5. Returns Deterministic Feature Tensor (TransformOutput)
    """
    def __init__(self, 
                 config_path: str = "configs/transform_config_v1.json"):
        self.macro_loader = MacroLoader()
        self.market_loader = MarketLoader()
        self.lag_aligner = LagAligner()
        self.corp_validator = CorporateActionValidator()
        self.transform_graph = TransformGraph(config_path)
        
        # Register standard nodes
        self.transform_graph.register(LogReturnNormalizer)
        self.transform_graph.register(ZScoreNormalizer)
        self.transform_graph.register(MinMaxNormalizer)
        self.transform_graph.register(DirectionalChangeDetector)
        
        self.transform_graph.build()

    @timed("preprocessing", "process_snapshot")
    def process_snapshot(self, 
                         market_source_path: str, 
                         macro_source_path: str, 
                         snapshot_id: str,
                         corporate_action_path: Optional[str] = None,
                         cutoff_date: Optional[str] = None) -> TransformOutput:
        """
        Processes a unified snapshot.
        snapshot_id tracks data provenance (Section 5.5).
        If cutoff_date is provided, applies event-time cutoff for replay.
        """
        logger.info(f"Loading snapshot: {snapshot_id}")
        
        # 1. Load Data
        market_df = self.market_loader.load(market_source_path, snapshot_id)
        macro_df = self.macro_loader.load(macro_source_path, snapshot_id)
        
        # Apply cutoff for replay (Section 5.5 Event-time playback mode)
        if cutoff_date and not market_df.empty:
            cutoff_ts = pd.to_datetime(cutoff_date, utc=True)
            market_df['timestamp'] = pd.to_datetime(market_df['timestamp'], utc=True)
            market_df = market_df[market_df["timestamp"] <= cutoff_ts]
            
        if cutoff_date and not macro_df.empty:
            cutoff_ts = pd.to_datetime(cutoff_date, utc=True)
            macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], utc=True)
            macro_df = macro_df[macro_df["timestamp"] <= cutoff_ts]
        
        # 2. Back-adjust splits/dividends if corp actions are provided
        if corporate_action_path:
            ca_df = pd.read_parquet(corporate_action_path) if str(corporate_action_path).endswith('.parquet') else pd.DataFrame()
            if cutoff_date and not ca_df.empty and 'ex_date' in ca_df.columns:
                ca_df['ex_date'] = pd.to_datetime(ca_df['ex_date'], utc=True)
                ca_df = ca_df[ca_df["ex_date"] <= cutoff_ts]
            market_df = self.corp_validator.apply_adjustments(market_df, ca_df)
            
        # 3. Align the slow variables backward in time to prevent lookahead
        aligned_df = self.lag_aligner.align(market_df, macro_df)
        
        # 4. Apply configured Transformations
        feature_df = self.transform_graph.execute(aligned_df)
        
        # 5. Build reproducible output artifact
        return self._build_deterministic_output(feature_df, snapshot_id)

    @timed("preprocessing", "replay_snapshot")
    def replay_snapshot(self,
                        market_source_path: str,
                        macro_source_path: str,
                        snapshot_id: str,
                        cutoff_date: str,
                        corporate_action_path: Optional[str] = None) -> TransformOutput:
        """
        Section 5.5 Replay Support. Reconstructs a snapshot identically as it would have appeared
        at the cutoff_date, filtering out any data mathematically published or timestamped after.
        """
        logger.info(f"Replaying snapshot: {snapshot_id} at {cutoff_date}")
        return self.process_snapshot(
            market_source_path=market_source_path,
            macro_source_path=macro_source_path,
            snapshot_id=snapshot_id,
            corporate_action_path=corporate_action_path,
            cutoff_date=cutoff_date
        )

    def _build_deterministic_output(self, df: pd.DataFrame, snapshot_id: str) -> TransformOutput:
        """
        Produces the SHA-256 hash guaranteeing the identical snapshot/config pairing always results in the same artifact.
        """
        return ReproducibilityHasher.build_deterministic_output(df, snapshot_id, "v1.0")
