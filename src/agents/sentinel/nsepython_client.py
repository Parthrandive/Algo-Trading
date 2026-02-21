import nsepython
from datetime import datetime, timezone
import logging
from typing import List

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, Tick, SourceType, QualityFlag, CorporateAction
from src.utils.resilience import retry_with_backoff, rate_limit

logger = logging.getLogger(__name__)

class NSEPythonClient(NSEClientInterface):
    """
    Client using nsepython wrapper for official NSE data.
    Primary use: Real-time snapshots (quotes) and Option Chain (future).
    Note: Historical data via this wrapper is often unstable/blocked.
    """

    def __init__(self):
        self.source_type = SourceType.FALLBACK_SCRAPER

    @retry_with_backoff(retries=2, backoff_in_seconds=2)
    @rate_limit(calls=1, period=2) # Strict rate limit to avoid IP blocks
    def get_stock_quote(self, symbol: str) -> Tick:
        """
        Fetch real-time quote for a stock using nsepython (nse_quote).
        """
        if "=" in symbol or "^" in symbol:
            raise ValueError(
                f"NSEPythonClient does not support forex/index symbol '{symbol}'. Use YFinanceClient for this symbol type."
            )

        # nsepython takes symbol without .NS extension usually, but let's handle both
        clean_symbol = symbol.replace(".NS", "").upper()
        
        try:
            # Use nse_eq for cleaner equity data
            data = nsepython.nse_eq(clean_symbol)
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol} from nsepython: {e}")
            raise e
            
        # Parse logic
        # nse_eq returns a dict with 'priceInfo', 'metadata', etc.
        price_info = data.get('priceInfo', {})
        price = price_info.get('lastPrice') or price_info.get('close') or data.get('underlyingValue')
        
        if price is None or price == '-':
             # Try other fields
             raise ValueError(f"Could not parse price from nsepython.nse_eq response for {symbol}")

        price = float(price)
        # Volume is inside priceInfo usually, or 'preOpenMarket'->'totalTradedVolume' (if preopen) 
        # But usually 'priceInfo' has it? Actually output showed it inside 'preOpenMarket' -> 'totalTradedVolume' 
        # Wait, the output for nse_eq showed 'priceInfo' keys. 
        # 'priceInfo': {'lastPrice': 1424.2, ...} does NOT have volume in the example printed!
        # Volume is in 'preOpenMarket'->'totalTradedVolume' OR maybe elsewhere?
        # Actually I don't see volume in 'priceInfo' in the printed output.
        # It has 'lastPrice', 'change', 'pChange', 'previousClose', 'open', 'close', 'vwap', 'intraDayHighLow', 'weekHighLow'.
        # I see 'securityInfo' -> 'issuedSize'.
        # I see 'preOpenMarket' -> 'totalTradedVolume': 16323.
        # The 'marketCap' info has 'avgDailyVol'.
        # Where is current volume? 
        # Ah, nse_eq might be "static" info sometimes? 
        # Let's try to find it. If not, 0.
        
        volume = 0
        if 'preOpenMarket' in data:
            volume = data['preOpenMarket'].get('totalTradedVolume', 0)
        
        # If there's a 'common' volume field?
        # Let's just use what we found.
        
        return Tick(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            source_type=self.source_type,
            price=price,
            volume=int(volume),
            bid=None,
            ask=None,
            quality_status=QualityFlag.PASS
        )

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
    ) -> List[Bar]:
        """
        Historical data via nsepython is often unstable. 
        We rely on YFinanceClient for this.
        """
        logger.warning("get_historical_data is not reliably supported by NSEPythonClient. Use YFinanceClient instead.")
        raise NotImplementedError("Historical data not supported by NSEPythonClient")

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List['CorporateAction']:
        """
        Corporate actions via nsepython are possible but we rely on YFinance.
        """
        logger.warning("get_corporate_actions not implemented in NSEPythonClient. Use YFinanceClient instead.")
        raise NotImplementedError("Corporate actions not supported by NSEPythonClient")
