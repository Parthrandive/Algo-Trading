from typing import Any, Dict, Type

from pydantic import BaseModel

from src.schemas.macro_data import MacroIndicator
from src.schemas.market_data import Bar, CorporateAction, Tick
from src.schemas.text_data import EarningsTranscript, NewsArticle, SocialPost

class SchemaRegistry:
    """
    Central registry for all data contracts.
    Enforces that data payloads match the expected schema version.
    """
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, version_key: str, model: Type[BaseModel]):
        """
        Register a model with a version key.
        Format: "DataType_vX.Y" (e.g., "Tick_v1.0")
        """
        if version_key in cls._registry:
            raise ValueError(f"Schema {version_key} already registered for {cls._registry[version_key]}")
        cls._registry[version_key] = model

    @classmethod
    def get_model(cls, version_key: str) -> Type[BaseModel]:
        if version_key not in cls._registry:
            raise ValueError(f"Schema {version_key} not found in registry.")
        return cls._registry[version_key]

    @classmethod
    def validate(cls, version_key: str, data: Dict[str, Any]) -> BaseModel:
        """
        Validate raw dict data against the registered schema.
        Raises ValidationError if mismatch.
        """
        model = cls.get_model(version_key)
        return model(**data)

def _register_default_schemas():
    # Pre-registering current versions (Auto-discovery could be added later).
    SchemaRegistry.register("Tick_v1.0", Tick)
    SchemaRegistry.register("Bar_v1.0", Bar)
    SchemaRegistry.register("CorporateAction_v1.0", CorporateAction)
    SchemaRegistry.register("MacroIndicator_v1.0", MacroIndicator)
    SchemaRegistry.register("MacroIndicator_v1.1", MacroIndicator)
    SchemaRegistry.register("NewsArticle_v1.0", NewsArticle)
    SchemaRegistry.register("SocialPost_v1.0", SocialPost)
    SchemaRegistry.register("EarningsTranscript_v1.0", EarningsTranscript)


_register_default_schemas()
