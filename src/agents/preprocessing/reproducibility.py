import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from src.schemas.preprocessing_data import TransformOutput

class ReproducibilityHasher:
    """
    Handles robust generation of SHA-256 output hashes given a deterministic pipeline state.
    Section 5.3 Deterministic hashing. Section 5.5 Event-time playback support.
    """
    
    @staticmethod
    def generate_hash(snapshot_id: str, records: List[Dict[str, Any]]) -> str:
        """
        Generates a SHA-256 hash using the snapshot ID and serialized records.
        """
        serialized_data = json.dumps(records, sort_keys=True)
        return hashlib.sha256(f"{snapshot_id}|{serialized_data}".encode("utf-8")).hexdigest()

    @staticmethod
    def build_deterministic_output(df: pd.DataFrame, snapshot_id: str, config_version: str = "v1.0") -> TransformOutput:
        """
        Produces the TransformOutput artifact, guaranteeing the identical snapshot/config 
        pairing always results in the same artifact.
        
        Args:
            df: The fully processed, normalized DataFrame
            snapshot_id: The provenance ID of the input snapshot
            config_version: The version string of the config used
        """
        # Sort values precisely before hashing to ensure determinism
        df = df.sort_values(by=["symbol", "timestamp"])
        df = df.fillna(0.0).infer_objects(copy=False) # Standardize representations of null
        
        # Dump robustly for the hashing
        records = df.to_dict(orient="records")
        for r in records:
             # Ensure dict serialization of datetimes handles JSON natively
             for k, v in r.items():
                 if isinstance(v, (pd.Timestamp, datetime)):
                     r[k] = v.isoformat()
                 elif pd.isna(v):  # Handle any remaining NaNs 
                     r[k] = None

        hash_digest = ReproducibilityHasher.generate_hash(snapshot_id, records)
        
        return TransformOutput(
            output_hash=hash_digest,
            input_snapshot_id=snapshot_id,
            transform_config_version=config_version,
            records=records
        )
