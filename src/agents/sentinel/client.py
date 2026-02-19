from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from src.schemas.market_data import Bar, Tick

class NSEClientInterface(ABC):
    @abstractmethod
    def get_stock_quote(self, symbol: str) -> Tick:
        """Fetch real-time quote for a stock."""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
    ) -> List[Bar]:
        """Fetch historical OHLCV data."""
        pass

import random
from typing import List
from datetime import datetime, timedelta, timezone
from src.schemas.market_data import Bar, Tick, SourceType

class MockNSEClient(NSEClientInterface):
    def get_stock_quote(self, symbol: str) -> Tick:
        """
        Returns a mock Tick with random price and volume.
        """
        price = round(random.uniform(100, 5000), 2)
        return Tick(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.MANUAL_OVERRIDE, # Mock data treated as override/test
            price=price,
            volume=random.randint(100, 10000),
            bid=round(price * 0.9995, 2),
            ask=round(price * 1.0005, 2)
        )

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
    ) -> List[Bar]:
        """
        Returns a list of mock Bars with random data for the given range.
        Assuming '1h' interval as per plan.
        """
        bars = []
        current_time = start_date
        while current_time <= end_date:
            open_price = round(random.uniform(100, 5000), 2)
            high_price = round(open_price * random.uniform(1.0, 1.02), 2)
            low_price = round(open_price * random.uniform(0.98, 1.0), 2)
            close_price = round(random.uniform(low_price, high_price), 2)
            
            bars.append(Bar(
                symbol=symbol,
                timestamp=current_time,
                source_type=SourceType.MANUAL_OVERRIDE,
                interval=interval,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.randint(1000, 50000)
            ))
            current_time += timedelta(hours=1)
        return bars
