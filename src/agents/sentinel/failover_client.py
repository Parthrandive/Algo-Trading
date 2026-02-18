import logging
import time
from datetime import datetime
from typing import List

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, Tick

logger = logging.getLogger(__name__)

class FailoverSentinelClient(NSEClientInterface):
    """
    Client that manages failover between a primary client and a list of fallback clients.
    """
    def __init__(
        self, 
        primary_client: NSEClientInterface, 
        fallback_clients: List[NSEClientInterface],
        failure_threshold: int = 3,
        cooldown_seconds: int = 60
    ):
        self.primary = primary_client
        self.fallbacks = fallback_clients
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        
        self._primary_failures = 0
        self._last_failure_time = 0
        self._is_primary_healthy = True

    def _check_health(self):
        """
        Check if primary should be considered healthy again after cooldown.
        """
        if not self._is_primary_healthy:
            if time.time() - self._last_failure_time > self.cooldown_seconds:
                logger.info("Primary client cooldown expired. Probing health...")
                # We optimistically mark it healthy. The next call will test it.
                # If it fails again, it will immediately become unhealthy.
                self._is_primary_healthy = True
                # Set failures to threshold - 1 so a single failure disables it again
                self._primary_failures = max(0, self.failure_threshold - 1)

    def _handle_primary_failure(self, e: Exception):
        self._primary_failures += 1
        self._last_failure_time = time.time()
        logger.warning(f"Primary client failed: {e}. Failure count: {self._primary_failures}")
        
        if self._primary_failures >= self.failure_threshold:
            if self._is_primary_healthy:
                 logger.error(f"Primary client crossed failure threshold ({self.failure_threshold}). Marking unhealthy.")
                 self._is_primary_healthy = False

    def get_stock_quote(self, symbol: str) -> Tick:
        self._check_health()

        if self._is_primary_healthy:
            try:
                return self.primary.get_stock_quote(symbol)
            except Exception as e:
                self._handle_primary_failure(e)
                # Fall through to fallbacks
        
        # Try fallbacks
        for client in self.fallbacks:
            try:
                return client.get_stock_quote(symbol)
            except Exception as e:
                logger.warning(f"Fallback client {client} failed: {e}")
                continue
        
        raise RuntimeError(f"All clients failed to fetch quote for {symbol}")

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Bar]:
        self._check_health()
        
        if self._is_primary_healthy:
            try:
                return self.primary.get_historical_data(symbol, start_date, end_date)
            except Exception as e:
                 self._handle_primary_failure(e)
        
        # Try fallbacks
        for client in self.fallbacks:
            try:
                # Note: Some clients might not support historical data (e.g. NSEPythonClient)
                # They should ideally return empty list or raise error. 
                # If they return empty list but log warning, we might want to skip them?
                # The interface returns List[Bar].
                data = client.get_historical_data(symbol, start_date, end_date)
                if not data:
                     # Could be valid no data, or unsupported. 
                     # For now we assume if it returns, it succeeded.
                     pass
                return data
            except Exception as e:
                logger.warning(f"Fallback client {client} failed history fetch: {e}")
                continue
        
        raise RuntimeError(f"All clients failed to fetch historical data for {symbol}")
