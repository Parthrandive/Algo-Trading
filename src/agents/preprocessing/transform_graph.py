from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.schemas.preprocessing_data import TransformConfig

class TransformNode(ABC):
    """
    Abstract base class for deterministic transformation nodes.
    Each node must define its versions and mathematical transform securely.
    """
    def __init__(self, config: TransformConfig):
        self.name = config.transform_name
        self.version = config.version
        self.input_schema_version = config.input_schema_version
        self.output_schema_version = config.output_schema_version
        self.parameters = config.parameters
        
        # Provenance enforcement 
        self.expected_version = self.get_expected_version()
        if self.version != self.expected_version:
            raise ValueError(
                f"Version mismatch: Config specifies v{self.version}, "
                f"but node {self.__class__.__name__} is v{self.expected_version}"
            )

    @classmethod
    @abstractmethod
    def get_expected_version(cls) -> str:
        """Returns the hardcoded version of this implementation."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the deterministic mathematical transformation."""
        pass
        
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper to evaluate schema versions before transforming."""
        # Note: Extensive schema enforcement happens at the Dag/Output boundaries.
        result_df = self.transform(df.copy(deep=True))
        
        # Attach provenance marker (schema versioning of the node)
        result_df.attrs["transform_provenance"] = result_df.attrs.get("transform_provenance", [])
        result_df.attrs["transform_provenance"].append({
            "node": self.name,
            "version": self.version,
            "output_schema_version": self.output_schema_version
        })
        
        return result_df

class TransformGraph:
    """
    Directed Acyclic Graph (DAG) for executing Preprocessing steps deterministically.
    Config-driven based on transform_config.json
    """
    def __init__(self, config_path: str = "configs/transform_config_v1.json"):
        self.nodes: List[TransformNode] = []
        self._registry: Dict[str, type] = {}
        
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                self.config_payload = json.load(f)
        else:
            self.config_payload = {"transforms": []}

    def register(self, node_class: type):
        """Register a node class logic for the internal DAG registry."""
        self._registry[node_class.__name__] = node_class
        
    def build(self):
        """Constructs the sequence of transforms based on the loaded config."""
        self.nodes = []
        for transform_def in self.config_payload.get("transforms", []):
            conf = TransformConfig(**transform_def)
            node_class_name = conf.transform_name
            
            if node_class_name not in self._registry:
                raise ValueError(f"TransformNode {node_class_name} is not registered in the graph.")
                
            node_class = self._registry[node_class_name]
            node_instance = node_class(config=conf)
            self.nodes.append(node_instance)
            
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the pipeline sequentially."""
        if not self.nodes:
            return df
            
        current_df = df
        for node in self.nodes:
            current_df = node.execute(current_df)
            
        return current_df
