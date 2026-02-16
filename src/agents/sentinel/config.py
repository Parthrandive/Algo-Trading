from pydantic import BaseModel, Field
from typing import List, Optional

class DataSourceConfig(BaseModel):
    name: str
    priority: int # 1 is highest
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class SentinelConfig(BaseModel):
    sources: List[DataSourceConfig]
    polling_interval_seconds: int = 60
    max_retries: int = 3
