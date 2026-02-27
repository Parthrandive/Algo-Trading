from typing import List, Dict, Any, Optional
import hashlib
import json
import logging

import pandas as pd

from src.agents.preprocessing.loader import MacroLoader, MarketLoader
from src.agents.preprocessing.lag_alignment import LagAligner, CorporateActionValidator
from src.agents.preprocessing.transform_graph import TransformGraph
from src.schemas.preprocessing_data import TransformOutput
from src.agents.preprocessing.reproducibility import ReproducibilityHasher

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
        self.transform_graph.build()

    def process_snapshot(self, 
                         market_source_path: str, 
                         macro_source_path: str, 
                         snapshot_id: str,
                         corporate_action_path: Optional[str] = None) -> TransformOutput:
        """
        Processes a unified snapshot.
        snapshot_id tracks data provenance (Section 5.5).
        """
        logger.info(f"Loading snapshot: {snapshot_id}")
        
        # 1. Load Data
        market_df = self.market_loader.load(market_source_path, snapshot_id)
        macro_df = self.macro_loader.load(macro_source_path, snapshot_id)
        
        # 2. Back-adjust splits/dividends if corp actions are provided
        if corporate_action_path:
            ca_df = pd.read_parquet(corporate_action_path) if str(corporate_action_path).endswith('.parquet') else pd.DataFrame()
            market_df = self.corp_validator.apply_adjustments(market_df, ca_df)
            
        # 3. Align the slow variables backward in time to prevent lookahead
        aligned_df = self.lag_aligner.align(market_df, macro_df)
        
        # 4. Apply configured Transformations
        feature_df = self.transform_graph.execute(aligned_df)
        
        # 5. Build reproducible output artifact
        return self._build_deterministic_output(feature_df, snapshot_id)

    def _build_deterministic_output(self, df: pd.DataFrame, snapshot_id: str) -> TransformOutput:
        """
        Produces the SHA-256 hash guaranteeing the identical snapshot/config pairing always results in the same artifact.
        """
        return ReproducibilityHasher.build_deterministic_output(df, snapshot_id, "v1.0")
