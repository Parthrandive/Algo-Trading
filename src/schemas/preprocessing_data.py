from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field

class FieldSpec(BaseModel):
    name: str
    data_type: str
    constraints: Dict[str, Any] = Field(default_factory=dict)

class PreprocessingContract(BaseModel):
    accepted_input_schemas: List[str]
    output_schema_version: str
    input_field_specs: List[FieldSpec]
    output_field_specs: List[FieldSpec]

class TransformConfig(BaseModel):
    transform_name: str
    version: str
    input_schema_version: str
    output_schema_version: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class TransformOutput(BaseModel):
    output_hash: str
    input_snapshot_id: str
    transform_config_version: str
    records: List[Dict[str, Any]]
    
    model_config = ConfigDict(frozen=True, extra="forbid")
