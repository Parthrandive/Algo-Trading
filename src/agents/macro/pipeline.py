"""
Macro Ingestion Pipeline orchestrating Bronze (raw) -> Silver (normalized) flow.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, cast

from src.agents.macro.client import DateRange, MacroClientInterface
from src.agents.macro.parsers import BaseParser
from src.agents.macro.recorder import MacroSilverRecorder
from src.agents.sentinel.failover_client import DegradationState
from src.schemas.macro_data import MacroIndicator, MacroIndicatorType

logger = logging.getLogger(__name__)


class MacroIngestPipeline:
    """
    Orchestrates the macro data ingestion flow.
    
    Responsibilities:
    1. Connect client output (Bronze) to a parser.
    2. Convert raw payloads into standard MacroIndicator records.
    3. Save records to the Silver layer (Parquet + DB).
    4. Guard feed integrity: if a feed fails violently, emit a 
       `reduce-only` degradation advisory.
    """

    def __init__(
        self,
        recorder: MacroSilverRecorder,
    ):
        self.recorder = recorder
        self._degradation_state = DegradationState.NORMAL

    @property
    def degradation_state(self) -> DegradationState:
        """
        Reflects feed health status.
        Transitions to REDUCE_ONLY on feed integrity failures.
        """
        return self._degradation_state

    def run_ingest(
        self,
        client: MacroClientInterface,
        indicator: MacroIndicatorType,
        date_range: DateRange,
        parser: BaseParser,
    ) -> Sequence[MacroIndicator]:
        """
        Execute ingestion for a single indicator.
        
        Note: The actual `get_indicator` signatures currently return Sequence[MacroIndicator],
        acting as both client + simplistic parser for some stubs. In a real-world Bronze 
        adapter, `get_indicator` would return generic `dict[str, Any]` which `BaseParser` 
        then normalizes.
        
        To support both our stubs and JSON parsers, this pipeline attempts to use the parser 
        if raw data is a dict, or proxies through if the client already returned records.
        """
        logger.info("Starting ingest for %s: %s", indicator.value, date_range)
        
        try:
            # Bronze layer fetch:
            raw_payloads = client.get_indicator(indicator, date_range)
            
            # Bronze -> Silver normalization
            # Handling stub clients vs real fetched dicts
            normalized_records: list[MacroIndicator] = []
            
            # Check if client returns raw dicts or already-normalized stub objects
            for item in raw_payloads:
                if isinstance(item, MacroIndicator):
                    normalized_records.append(item)
                elif isinstance(item, dict):
                    # Actual parser routing
                    records = parser.parse(cast(dict[str, Any], item))
                    normalized_records.extend(records)
                else:
                    raise TypeError(f"Unexpected payload type from client: {type(item)}")
                    
            if not normalized_records:
                logger.warning("No valid records produced for %s", indicator.value)
                return []
                
            # Silver layer persist
            self.recorder.save_indicators(normalized_records)
            
            # Reset advisory on successful run
            if self._degradation_state != DegradationState.NORMAL:
                logger.info("Feed %s recovered. Returning to normal state.", indicator.value)
                self._degradation_state = DegradationState.NORMAL
                
            return normalized_records

        except Exception as e:
            # Feed Integrity Failure triggers reduce-only mode.
            logger.error("Feed integrity failure during %s ingestion: %s", indicator.value, e)
            self._degradation_state = DegradationState.REDUCE_ONLY
            raise
