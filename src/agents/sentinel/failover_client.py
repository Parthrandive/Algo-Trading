import logging
import time
from datetime import datetime
from enum import Enum
from typing import List

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, QualityFlag, SourceType, Tick, CorporateAction

logger = logging.getLogger(__name__)


class DegradationState(str, Enum):
    NORMAL = "normal"
    REDUCE_ONLY = "reduce-only"
    CLOSE_ONLY_ADVISORY = "close-only advisory"


class FailoverSentinelClient(NSEClientInterface):
    """
    Manages primary/fallback failover and feed degradation states.
    """

    def __init__(
        self,
        primary_client: NSEClientInterface,
        fallback_clients: List[NSEClientInterface],
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
        recovery_success_threshold: int = 2,
        fallback_source_type: SourceType = SourceType.FALLBACK_SCRAPER,
    ):
        self.primary = primary_client
        self.fallbacks = fallback_clients
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.recovery_success_threshold = recovery_success_threshold
        self.fallback_source_type = fallback_source_type

        self._primary_failures = 0
        self._last_failure_time = 0
        self._is_primary_healthy = True
        self._degradation_state = DegradationState.NORMAL
        self._primary_recovery_successes = 0

    def _check_health(self):
        """
        Check if primary should be considered healthy again after cooldown window.
        """
        if not self._is_primary_healthy:
            if time.time() - self._last_failure_time > self.cooldown_seconds:
                logger.info("Primary client cooldown expired. Probing health...")
                self._is_primary_healthy = True
                self._primary_failures = max(0, self.failure_threshold - 1)
                self._primary_recovery_successes = 0

    def _handle_primary_failure(self, e: Exception):
        self._primary_recovery_successes = 0
        self._primary_failures += 1
        self._last_failure_time = time.time()
        logger.warning(f"Primary client failed: {e}. Failure count: {self._primary_failures}")

        if self._primary_failures >= self.failure_threshold:
            if self._is_primary_healthy:
                logger.error(f"Primary client crossed failure threshold ({self.failure_threshold}). Marking unhealthy.")
                self._is_primary_healthy = False
            self._degradation_state = DegradationState.REDUCE_ONLY

    def _handle_primary_success(self):
        if self._degradation_state == DegradationState.NORMAL:
            self._primary_failures = 0
            return

        self._primary_recovery_successes += 1
        if self._primary_recovery_successes >= self.recovery_success_threshold:
            logger.info("Primary source stabilized. Returning to normal state.")
            self._degradation_state = DegradationState.NORMAL
            self._is_primary_healthy = True
            self._primary_failures = 0
            self._primary_recovery_successes = 0

    def _handle_fallback_success(self):
        self._degradation_state = DegradationState.REDUCE_ONLY

    def _handle_all_sources_failure(self):
        self._degradation_state = DegradationState.CLOSE_ONLY_ADVISORY

    @property
    def degradation_state(self) -> DegradationState:
        return self._degradation_state

    def _tag_tick_as_fallback(self, tick: Tick) -> Tick:
        return tick.model_copy(
            update={
                "source_type": self.fallback_source_type,
                "quality_status": QualityFlag.WARN,
            }
        )

    def _tag_bars_as_fallback(self, bars: List[Bar]) -> List[Bar]:
        return [
            bar.model_copy(
                update={
                    "source_type": self.fallback_source_type,
                    "quality_status": QualityFlag.WARN,
                }
            )
            for bar in bars
        ]

    def _tag_actions_as_fallback(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        return [
            action.model_copy(
                update={
                    "source_type": self.fallback_source_type,
                    "quality_status": QualityFlag.WARN,
                }
            )
            for action in actions
        ]

    def get_stock_quote(self, symbol: str) -> Tick:
        self._check_health()

        if self._is_primary_healthy:
            try:
                tick = self.primary.get_stock_quote(symbol)
                self._handle_primary_success()
                return tick
            except Exception as e:
                self._handle_primary_failure(e)

        for client in self.fallbacks:
            try:
                tick = client.get_stock_quote(symbol)
                self._handle_fallback_success()
                return self._tag_tick_as_fallback(tick)
            except Exception as e:
                logger.warning(f"Fallback client {client} failed: {e}")
                continue

        self._handle_all_sources_failure()
        raise RuntimeError(f"All clients failed to fetch quote for {symbol}")

    @staticmethod
    def _fetch_history(
        client: NSEClientInterface,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> List[Bar]:
        try:
            return client.get_historical_data(symbol, start_date, end_date, interval=interval)
        except TypeError:
            # Backward compatibility for clients with the older interface.
            return client.get_historical_data(symbol, start_date, end_date)

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
    ) -> List[Bar]:
        self._check_health()

        if self._is_primary_healthy:
            try:
                data = self._fetch_history(self.primary, symbol, start_date, end_date, interval=interval)
                self._handle_primary_success()
                return data
            except NotImplementedError:
                pass  # Skip directly to fallback without marking primary as unhealthy
            except Exception as e:
                self._handle_primary_failure(e)

        for client in self.fallbacks:
            try:
                data = self._fetch_history(client, symbol, start_date, end_date, interval=interval)
                if not data:
                    logger.warning(f"Fallback client {client} returned no historical data for {symbol}")
                    continue
                self._handle_fallback_success()
                return self._tag_bars_as_fallback(data)
            except Exception as e:
                logger.warning(f"Fallback client {client} failed history fetch: {e}")
                continue

        self._handle_all_sources_failure()
        raise RuntimeError(f"All clients failed to fetch historical data for {symbol}")

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List['CorporateAction']:
        self._check_health()
        saw_empty_data = False

        if self._is_primary_healthy:
            try:
                data = self.primary.get_corporate_actions(symbol, start_date, end_date)
                self._handle_primary_success()
                if data:
                    return data
                saw_empty_data = True
                logger.warning(f"Primary client returned no corporate actions for {symbol}; probing fallbacks.")
            except NotImplementedError:
                pass  # Skip directly to fallback without marking primary as unhealthy
            except Exception as e:
                self._handle_primary_failure(e)

        for client in self.fallbacks:
            try:
                data = client.get_corporate_actions(symbol, start_date, end_date)
                if not data:
                    saw_empty_data = True
                    logger.warning(f"Fallback client {client} returned no corporate actions for {symbol}")
                    continue
                self._handle_fallback_success()
                return self._tag_actions_as_fallback(data)
            except NotImplementedError:
                continue
            except Exception as e:
                logger.warning(f"Fallback client {client} failed corporate actions fetch: {e}")
                continue

        if saw_empty_data:
            return []

        self._handle_all_sources_failure()
        raise RuntimeError(f"All clients failed to fetch corporate actions for {symbol}")
