import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional, Sequence, Tuple

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.agents.macro.client import DateRange, MacroClientInterface
from src.agents.macro.parsers import BaseParser
from src.agents.macro.pipeline import MacroIngestPipeline
from src.db.connection import get_engine, get_session
from src.db.models import IngestionLog, MacroIndicatorDB
from src.schemas.macro_data import MacroIndicator, MacroIndicatorType

logger = logging.getLogger(__name__)

class MacroScheduler:
    """
    Scheduler for Macro Ingestion jobs.
    Reads runtime config for retry/backoff policies.
    """

    def __init__(
        self,
        config_path: str,
        pipeline: MacroIngestPipeline,
        registry: Dict[MacroIndicatorType, Tuple[MacroClientInterface, BaseParser]],
        database_url: Optional[str] = None,
    ):
        self.config_path = config_path
        self.pipeline = pipeline
        self.registry = registry
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.engine = get_engine(database_url)
        self.Session = get_session(self.engine)

    def get_retry_policy(self, indicator: MacroIndicatorType) -> dict[str, Any]:
        """Extract retry policy from config for the indicator, or fallback to default."""
        ind_config = self.config.get("indicator_configs", {}).get(indicator.value, {})
        
        # If there are sources, use the first source's retry policy (naive approach)
        sources = ind_config.get("sources", [])
        if sources and "retry" in sources[0]:
            return sources[0]["retry"]
            
        return self.config.get("default_request_policy", {}).get(
            "retry", {"max_attempts": 3, "base_backoff_seconds": 2}
        )

    def is_already_ingested(self, indicator: MacroIndicatorType, date_range: DateRange) -> bool:
        """
        Idempotent ingest guard:
        Checks if records for the given indicator and date range already exist in the DB.
        """
        try:
            with self.Session() as session:
                count = (
                    session.query(MacroIndicatorDB)
                    .filter(
                        MacroIndicatorDB.indicator_name == indicator.value,
                        MacroIndicatorDB.timestamp >= date_range.start,
                        MacroIndicatorDB.timestamp <= date_range.end,
                    )
                    .count()
                )
                return count > 0
        except Exception as e:
            logger.warning("Failed to check ingestion status (defaulting to False): %s", e)
            return False

    def log_provenance(
        self,
        indicator: MacroIndicatorType,
        records_ingested: int,
        status: str,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        dataset_snapshot_id: Optional[str] = None,
    ) -> None:
        """Write provenance logging to IngestionLog table."""
        try:
            with self.Session() as session:
                session.execute(
                    insert(IngestionLog).values(
                        run_timestamp=datetime.now(UTC),
                        symbol=indicator.value,
                        data_type="macro",
                        records_ingested=records_ingested,
                        status=status,
                        error_message=error_message,
                        duration_ms=duration_ms,
                        dataset_snapshot_id=dataset_snapshot_id,
                        code_hash="macro_v1.1", # placeholder for actual commit hash
                    )
                )
                session.commit()
        except Exception as e:
            logger.error("Failed to write to IngestionLog: %s", e)

    def run_job(self, indicator: MacroIndicatorType, date_range: DateRange) -> Sequence[MacroIndicator]:
        """
        Run ingestion job for a specific indicator.
        Applies exponential backoff retries and idempotent guards.
        """
        start_time = time.time()
        
        if indicator not in self.registry:
            logger.error("No client/parser registered for %s", indicator.value)
            return []
            
        client, parser = self.registry[indicator]

        # Idempotent ingest guard
        if self.is_already_ingested(indicator, date_range):
            logger.info("Skipping ingestion for %s; already exists in date range.", indicator.value)
            return []

        retry_policy = self.get_retry_policy(indicator)
        max_attempts = retry_policy.get("max_attempts", 3)
        base_backoff = retry_policy.get("base_backoff_seconds", 2)

        attempt = 1
        last_exception = None
        
        while attempt <= max_attempts:
            try:
                logger.info("Running job for %s (Attempt %d/%d)", indicator.value, attempt, max_attempts)
                records = self.pipeline.run_ingest(client, indicator, date_range, parser)
                
                duration_ms = (time.time() - start_time) * 1000
                
                # We can construct a simplistic snapshot ID
                snapshot_id = f"macro_snapshot_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
                
                self.log_provenance(
                    indicator=indicator,
                    records_ingested=len(records),
                    status="success",
                    duration_ms=duration_ms,
                    dataset_snapshot_id=snapshot_id,
                )
                return records
                
            except Exception as e:
                last_exception = e
                logger.warning("Ingestion failed for %s on attempt %d: %s", indicator.value, attempt, e)
                if attempt < max_attempts:
                    sleep_time = base_backoff * (2 ** (attempt - 1))
                    logger.info("Sleeping for %d seconds before retry", sleep_time)
                    time.sleep(sleep_time)
                attempt += 1

        # If it reaches here, all attempts failed
        duration_ms = (time.time() - start_time) * 1000
        logger.error("All %d attempts failed for %s", max_attempts, indicator.value)
        self.log_provenance(
            indicator=indicator,
            records_ingested=0,
            status="failed",
            error_message=str(last_exception),
            duration_ms=duration_ms,
        )
        raise RuntimeError(f"Ingestion failed for {indicator.value} after {max_attempts} attempts") from last_exception
