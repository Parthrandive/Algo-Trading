from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol, List

from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.client import NSEClientInterface
from src.agents.sentinel.config import SessionRules
from src.schemas.market_data import Bar, Tick, CorporateAction
from src.utils.latency import timed


class SilverRecorderProtocol(Protocol):
    def save_bars(self, bars: List[Bar]) -> None: ...
    def save_ticks(self, ticks: List[Tick]) -> None: ...
    def save_corporate_actions(self, actions: List[CorporateAction]) -> None: ...


class SentinelIngestPipeline:
    """
    Bronze -> Silver ingest pipeline for NSE Sentinel.
    """

    def __init__(
        self,
        client: NSEClientInterface,
        silver_recorder: SilverRecorderProtocol,
        bronze_recorder: Optional[BronzeRecorder] = None,
        session_rules: Optional[SessionRules] = None,
    ):
        self.client = client
        self.silver_recorder = silver_recorder
        self.bronze_recorder = bronze_recorder
        self.session_rules = session_rules

    def _within_session(self, timestamp: datetime) -> bool:
        if self.session_rules is None:
            return True
        return self.session_rules.is_trading_session(timestamp)

    @timed("sentinel", "ingest_quote")
    def ingest_quote(self, symbol: str, schema_id: str = "market.tick.v1") -> Tick:
        tick = self.client.get_stock_quote(symbol)
        if self.bronze_recorder is not None:
            self.bronze_recorder.save_event(
                source_id=tick.source_type.value,
                payload=tick.model_dump(mode="json"),
                event_time=tick.timestamp,
                symbol=symbol,
                schema_id=schema_id,
            )
        self.silver_recorder.save_ticks([tick])
        return tick

    @timed("sentinel", "ingest_historical")
    def ingest_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
        schema_id: str = "market.bar.v1",
    ) -> list[Bar]:
        bars = self.client.get_historical_data(symbol, start_date, end_date, interval=interval)
        accepted_bars: list[Bar] = []
        for bar in bars:
            if self.bronze_recorder is not None:
                self.bronze_recorder.save_event(
                    source_id=bar.source_type.value,
                    payload=bar.model_dump(mode="json"),
                    event_time=bar.timestamp,
                    symbol=symbol,
                    schema_id=schema_id,
                )
            if self._within_session(bar.timestamp):
                accepted_bars.append(bar)

        self.silver_recorder.save_bars(accepted_bars)
        return accepted_bars

    @timed("sentinel", "ingest_corporate_actions")
    def ingest_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        schema_id: str = "market.corporate_action.v1",
    ) -> list[CorporateAction]:
        actions = self.client.get_corporate_actions(symbol, start_date, end_date)

        for action in actions:
            if self.bronze_recorder is not None:
                self.bronze_recorder.save_event(
                    source_id=action.source_type.value,
                    payload=action.model_dump(mode="json"),
                    event_time=action.ex_date,
                    symbol=action.symbol,
                    schema_id=schema_id,
                )

        self.silver_recorder.save_corporate_actions(actions)
        return actions
