import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
import logging

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, Tick, SourceType, QualityFlag
from src.utils.resilience import retry_with_backoff, rate_limit

logger = logging.getLogger(__name__)

class YFinanceClient(NSEClientInterface):
    """
    Primary connector using yfinance to fetch market data.
    """
    
    def __init__(self):
        self.source_type = SourceType.OFFICIAL_API # Using yfinance as 'official' proxy for now per plan

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    @rate_limit(calls=2, period=1) # Limit to 2 calls per second to be safe
    def get_stock_quote(self, symbol: str) -> Tick:
        """
        Fetch real-time quote for a stock using yfinance.
        Note: yfinance real-time data might be delayed.
        """
        ticker = yf.Ticker(symbol)
        # fast_info is often faster/more reliable for current price than .info
        info = ticker.fast_info
        
        if info is None or info.last_price is None:
             # Fallback to .info if fast_info fails or is empty, though less preferred
             info_dict = ticker.info
             current_price = info_dict.get('currentPrice') or info_dict.get('regularMarketPrice')
             if current_price is None:
                 raise ValueError(f"Could not fetch quote for {symbol}")
             price = float(current_price)
             volume = info_dict.get('volume', 0)
        else:
             price = float(info.last_price)
             volume = int(info.last_volume) if info.last_volume else 0

        # Create Tick object
        # Note: yfinance might not give bid/ask easily in fast_info without market status
        # We will leave bid/ask None for now or try to fetch if critical
        
        return Tick(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            source_type=self.source_type,
            price=price,
            volume=volume,
            bid=None, # Not always available
            ask=None,
            quality_status=QualityFlag.PASS
        )

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    @rate_limit(calls=1, period=1) # Heavier call, 1 per second
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1h") -> List[Bar]:
        """
        Fetch historical OHLCV data.
        """
        ticker = yf.Ticker(symbol)
        
        # yfinance expects YYYY-MM-DD string or datetime
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
            return []
            
        bars = []
        for index, row in df.iterrows():
            # index is DatetimeIndex, usually timezone aware if using yfinance
            ts = index.to_pydatetime()
            if ts.tzinfo is None:
                # If naive, assume UTC or market local? yfinance usually returns market local (IST for NSE)
                # But best to ensure we handle it.
                # For now, let's assume it matches the requested or is localized.
                # If naive, we might need to localize.
                pass
            
            # Map to Bar
            bar = Bar(
                symbol=symbol,
                timestamp=ts,
                source_type=self.source_type,
                interval=interval,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=int(row['Volume']),
                quality_status=QualityFlag.PASS
            )
            bars.append(bar)
            
        return bars
