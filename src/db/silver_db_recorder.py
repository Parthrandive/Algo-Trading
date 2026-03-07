import logging
from typing import List
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert
from src.schemas.market_data import Bar, CorporateAction, Tick
from src.schemas.macro_data import MacroIndicator
from src.schemas.text_data import NewsArticle, SocialPost, EarningsTranscript, TextDataBase
from src.utils.validation import StreamMonotonicityChecker
from src.db.connection import get_engine, get_session
from src.db.models import OHLCVBar, CorporateActionDB, TickData, MacroIndicatorDB, TextItemDB, QuarantineBar, IngestionLog

logger = logging.getLogger(__name__)
TEXT_ITEM_COLUMNS = {column.name for column in TextItemDB.__table__.columns}

class SilverDBRecorder:
    """
    Database-backed Silver Layer recorder.
    Drop-in replacement for SilverRecorder.
    """
    def __init__(self, database_url: str = None):
        self.engine = get_engine(database_url)
        self.Session = get_session(self.engine)
        self.monotonicity_checker = StreamMonotonicityChecker()

    def save_bars(self, bars: List[Bar]) -> None:
        if not bars:
            return

        valid_bars = []
        quarantined_bars = []
        
        # Sort bars by timestamp to ensure we process in order within the batch
        sorted_bars = sorted(bars, key=lambda b: b.timestamp)

        for bar in sorted_bars:
            if self.monotonicity_checker.check(bar.symbol, bar.timestamp, bar.interval):
                valid_bars.append(bar)
            else:
                logger.warning(f"Quarantining out-of-order bar for {bar.symbol} at {bar.timestamp}")
                quarantined_bars.append(bar)

        with self.Session() as session:
            try:
                if valid_bars:
                    # Use PostgreSQL INSERT ... ON CONFLICT DO UPDATE
                    stmt = insert(OHLCVBar).values([b.model_dump(by_alias=True) for b in valid_bars])
                    
                    update_dict = {
                        c.name: c for c in stmt.excluded 
                        if not c.primary_key
                    }
                    
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['timestamp', 'symbol', 'interval'],
                        set_=update_dict
                    )
                    session.execute(stmt)

                if quarantined_bars:
                    # Quarantined bars are inserted blindly
                    quarantine_data = [b.model_dump(by_alias=True) for b in quarantined_bars]
                    for q in quarantine_data:
                        q['reason'] = 'monotonicity_violation'
                        q['quarantined_at'] = datetime.now(timezone.utc)
                    session.execute(insert(QuarantineBar).values(quarantine_data).on_conflict_do_nothing())

                # Log the ingestion run
                session.execute(
                    insert(IngestionLog).values(
                        run_timestamp=datetime.now(timezone.utc),
                        data_type="ohlcv",
                        records_ingested=len(valid_bars),
                        records_quarantined=len(quarantined_bars),
                        status="success"
                    )
                )
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save bars to database: {e}")
                
                # Log failure
                try:
                    with self.Session() as fail_session:
                        fail_session.execute(
                            insert(IngestionLog).values(
                                run_timestamp=datetime.now(timezone.utc),
                                data_type="ohlcv",
                                records_ingested=0,
                                records_quarantined=len(quarantined_bars),
                                status="failed",
                                error_message=str(e)
                            )
                        )
                        fail_session.commit()
                except Exception as log_error:
                    logger.warning(f"Failed to write failure ingestion_log row: {log_error}")
                raise

    def save_corporate_actions(self, actions: List[CorporateAction]) -> None:
        if not actions:
            return

        with self.Session() as session:
            try:
                stmt = insert(CorporateActionDB).values([a.model_dump(by_alias=True) for a in actions])
                update_dict = {c.name: c for c in stmt.excluded if not c.primary_key}
                stmt = stmt.on_conflict_do_update(
                    index_elements=['ex_date', 'symbol', 'action_type'],
                    set_=update_dict
                )
                session.execute(stmt)
                
                session.execute(
                    insert(IngestionLog).values(
                        run_timestamp=datetime.now(timezone.utc),
                        data_type="corporate_action",
                        records_ingested=len(actions),
                        status="success"
                    )
                )
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save corporate actions: {e}")
                raise

    def save_ticks(self, ticks: List[Tick]) -> None:
        if not ticks:
            return

        with self.Session() as session:
            try:
                stmt = insert(TickData).values([t.model_dump(by_alias=True) for t in ticks])
                update_dict = {c.name: c for c in stmt.excluded if not c.primary_key}
                stmt = stmt.on_conflict_do_update(
                    index_elements=['timestamp', 'symbol'],
                    set_=update_dict
                )
                session.execute(stmt)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save ticks: {e}")
                raise

    def save_macro_indicators(self, indicators: List[MacroIndicator]) -> None:
        if not indicators:
            return
            
        with self.Session() as session:
            try:
                stmt = insert(MacroIndicatorDB).values([i.model_dump(by_alias=True) for i in indicators])
                update_dict = {c.name: c for c in stmt.excluded if not c.primary_key}
                stmt = stmt.on_conflict_do_update(
                    index_elements=['timestamp', 'indicator_name'],
                    set_=update_dict
                )
                session.execute(stmt)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save macro indicators: {e}")
                raise
                
    def save_text_items(self, items: List[TextDataBase]) -> None:
        if not items:
            return
            
        with self.Session() as session:
            try:
                # TextItems have different subclasses, we map them universally to the TextItemDB model
                items_data = []
                for item in items:
                    data = item.model_dump(by_alias=True)
                    if isinstance(item, NewsArticle):
                        data["item_type"] = "news"
                    elif isinstance(item, SocialPost):
                        data["item_type"] = "social"
                    elif isinstance(item, EarningsTranscript):
                        data["item_type"] = "transcript"
                    else:
                        data["item_type"] = "unknown"

                    # Text schemas may contain enrichment fields not present in DB.
                    filtered = {key: value for key, value in data.items() if key in TEXT_ITEM_COLUMNS}
                    
                    # Pad missing columns with None to prevent heterogeneous bulk insert failures
                    for col in TEXT_ITEM_COLUMNS:
                        if col not in filtered:
                            filtered[col] = None
                            
                    items_data.append(filtered)
                    
                stmt = insert(TextItemDB).values(items_data)
                update_dict = {c.name: c for c in stmt.excluded if not c.primary_key}
                stmt = stmt.on_conflict_do_update(
                    index_elements=['source_type', 'source_id'],
                    set_=update_dict
                )
                session.execute(stmt)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save text items: {e}")
                raise
