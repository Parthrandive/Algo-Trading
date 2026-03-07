import yfinance as yf
from datetime import datetime, timezone
import requests
from typing import List, Optional
import logging

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, Tick, SourceType, QualityFlag, CorporateAction, CorporateActionType
from src.utils.resilience import retry_with_backoff, rate_limit

logger = logging.getLogger(__name__)

class YFinanceClient(NSEClientInterface):
    """
    Primary connector using yfinance to fetch market data & corporate actions.
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

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    @rate_limit(calls=1, period=1)
    def get_corporate_actions(self, symbol: str, start_date: datetime, end_date: datetime) -> List[CorporateAction]:
        """
        Fetch historical corporate actions (Dividends and Stock Splits) using yfinance.
        """
        ticker = yf.Ticker(symbol)
        actions_df = ticker.actions
        
        if actions_df.empty:
            return []

        # Convert index to timezone-aware if naive (assuming local Market Time / IST for .NS)
        # yfinance usually returns tz-aware index for actions, but we should be robust
        if actions_df.index.tzinfo is None:
             # Just a fallback, might not be necessary depending on yf version
            actions_df.index = actions_df.index.tz_localize('Asia/Kolkata')
            
        # Filter by date range
        # Note: yfinance returns all history by default for actions.
        # Ensure we compare timezone-aware dates
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
            
        mask = (actions_df.index >= start_date) & (actions_df.index <= end_date)
        filtered_df = actions_df.loc[mask]
        
        corp_actions = []
        for dt, row in filtered_df.iterrows():
            ts = dt.to_pydatetime()
            
            # Check for Dividends
            if 'Dividends' in row and row['Dividends'] > 0:
                corp_actions.append(CorporateAction(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc), # When we ingested it
                    source_type=self.source_type,
                    action_type=CorporateActionType.DIVIDEND,
                    value=float(row['Dividends']),
                    ex_date=ts,
                    quality_status=QualityFlag.PASS
                ))
            
            # Check for Stock Splits
            if 'Stock Splits' in row and row['Stock Splits'] > 0:
                # yfinance reports stock splits as a float, e.g., 2.0 implies 2:1 ratio
                split_ratio = float(row['Stock Splits'])
                # Format to a readable string if you prefer, e.g. "2.0:1"
                ratio_str = f"{split_ratio}:1"
                
                corp_actions.append(CorporateAction(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    source_type=self.source_type,
                    action_type=CorporateActionType.SPLIT,
                    ratio=ratio_str,
                    ex_date=ts,
                    quality_status=QualityFlag.PASS
                ))
                
        return corp_actions
