import pandas as pd
import pytest

from src.agents.preprocessing.transform_graph import TransformGraph
from src.agents.preprocessing.normalizers import (
    LogReturnNormalizer, 
    ZScoreNormalizer, 
    DirectionalChangeDetector,
    MinMaxNormalizer
)
from src.schemas.preprocessing_data import TransformConfig

def test_transform_graph_registers_and_executes():
    # Setup inline config for deterministic runtime
    config = {
        "transforms": [
            {
                "transform_name": "LogReturnNormalizer",
                "version": "1.0",
                "input_schema_version": "market.bar.v1",
                "output_schema_version": "features.v1",
                "parameters": {"target_column": "close", "output_column": "returns"}
            },
            {
                "transform_name": "ZScoreNormalizer",
                "version": "1.0",
                "input_schema_version": "features.v1",
                "output_schema_version": "features.v2",
                "parameters": {"target_column": "returns", "output_column": "z_returns", "window": 2}
            }
        ]
    }
    
    graph = TransformGraph()
    graph.config_payload = config
    
    graph.register(LogReturnNormalizer)
    graph.register(ZScoreNormalizer)
    
    graph.build()
    assert len(graph.nodes) == 2
    
    df = pd.DataFrame({"close": [100.0, 110.0, 121.0]})
    # Log Returns:
    # 1: ln(1.1) ~ 0.0953
    # 2: ln(1.1) ~ 0.0953
    
    out_df = graph.execute(df)
    assert "returns" in out_df.columns
    assert "z_returns" in out_df.columns
    
    # Check that provenance headers propagated
    assert "transform_provenance" in out_df.attrs
    prov = out_df.attrs["transform_provenance"]
    assert len(prov) == 2
    assert prov[0]["node"] == "LogReturnNormalizer"
    assert prov[1]["node"] == "ZScoreNormalizer"

def test_transform_version_rejection():
    # Expecting failure if config dictates a version standard the node code does not match
    config = TransformConfig(
        transform_name="ZScoreNormalizer",
        version="2.0",  # Code represents 1.0!
        input_schema_version="v1",
        output_schema_version="v1"
    )
    
    with pytest.raises(ValueError, match="Version mismatch"):
        ZScoreNormalizer(config)

def test_directional_change_detector():
    df = pd.DataFrame({"value": [100.0, 101.0, 106.0, 100.0]})
    # Returns ~ 1%, 4.95% (near 5%), -5.6%
    
    config = TransformConfig(
        transform_name="DirectionalChangeDetector",
        version="1.0",
        input_schema_version="v1",
        output_schema_version="v1",
        parameters={"target_column": "value", "output_column": "flag", "threshold": 0.04}
    )
    node = DirectionalChangeDetector(config)
    
    out_df = node.execute(df)
    
    assert out_df["flag"].iloc[0] == 0  # Initial NaNs fallback
    assert out_df["flag"].iloc[1] == 0  # +1% < 4% change
    assert out_df["flag"].iloc[2] == 1  # + 5% change is > 4% change
    assert out_df["flag"].iloc[3] == -1 # -5.6% drop is <= -4% change
