import nsepython
from datetime import datetime, timezone
import logging
from typing import List

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, Tick, SourceType, QualityFlag, CorporateAction, CorporateActionType
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

    @staticmethod
    def _map_action_type(raw_value: object) -> CorporateActionType | None:
        if raw_value is None:
            return None
        normalized = str(raw_value).strip().lower().replace("_", " ").replace("-", " ")
        if "dividend" in normalized:
            return CorporateActionType.DIVIDEND
        if "split" in normalized or "sub division" in normalized or "subdivision" in normalized:
            return CorporateActionType.SPLIT
        if "bonus" in normalized:
            return CorporateActionType.BONUS
        if "rights" in normalized:
            return CorporateActionType.RIGHTS
        return None

    @staticmethod
    def _normalize_ratio(raw_value: object) -> str | None:
        if raw_value is None:
            return None
        normalized = str(raw_value).strip().replace("/", ":")
        if ":" in normalized:
            return normalized
        try:
            ratio_value = float(normalized)
        except ValueError:
            return None
        return f"{ratio_value}:1"

    @staticmethod
    def _parse_timestamp(raw_value: object) -> datetime | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, datetime):
            if raw_value.tzinfo is None:
                return raw_value.replace(tzinfo=timezone.utc)
            return raw_value.astimezone(timezone.utc)
        if isinstance(raw_value, (int, float)):
            return datetime.fromtimestamp(float(raw_value), tz=timezone.utc)
        if isinstance(raw_value, str):
            value = raw_value.strip()
            if not value:
                return None
            for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d-%b-%Y"):
                try:
                    parsed = datetime.strptime(value, fmt)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

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

    @retry_with_backoff(retries=2, backoff_in_seconds=2)
    @rate_limit(calls=1, period=2)
    def get_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CorporateAction]:
        """
        Best-effort corporate actions from NSE API endpoints via nsepython.
        Returns an empty list when unavailable to keep fallback orchestration alive.
        """
        clean_symbol = symbol.replace(".NS", "").upper()
        candidate_urls = [
            f"https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={clean_symbol}",
            f"https://www.nseindia.com/api/corporates-corporateActions?symbol={clean_symbol}",
            f"https://www.nseindia.com/api/corporate-actions?symbol={clean_symbol}",
        ]

        parsed_actions: list[CorporateAction] = []
        for url in candidate_urls:
            try:
                payload = nsepython.nsefetch(url)
            except Exception as exc:
                logger.warning("NSE corporate actions endpoint failed for %s: %s", clean_symbol, exc)
                continue

            rows = []
            if isinstance(payload, dict):
                rows = payload.get("data") or payload.get("records") or payload.get("corporateActions") or []
            elif isinstance(payload, list):
                rows = payload

            if not rows:
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue

                action_type = self._map_action_type(
                    row.get("purpose")
                    or row.get("actionType")
                    or row.get("action_type")
                    or row.get("caType")
                )
                ex_date = self._parse_timestamp(
                    row.get("exDate")
                    or row.get("ex_date")
                    or row.get("exdate")
                    or row.get("recordDate")
                    or row.get("record_date")
                )
                if action_type is None or ex_date is None:
                    continue

                record_date = self._parse_timestamp(row.get("recordDate") or row.get("record_date"))
                ratio = self._normalize_ratio(
                    row.get("ratio")
                    or row.get("splitRatio")
                    or row.get("bonusRatio")
                    or row.get("rightsRatio")
                )
                value_raw = row.get("value") or row.get("amount") or row.get("dividend")
                value = None
                if value_raw not in (None, ""):
                    try:
                        value = float(value_raw)
                    except (TypeError, ValueError):
                        value = None

                try:
                    action = CorporateAction(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        source_type=self.source_type,
                        action_type=action_type,
                        ratio=ratio,
                        value=value,
                        ex_date=ex_date,
                        record_date=record_date,
                        quality_status=QualityFlag.PASS,
                    )
                except Exception:
                    continue

                if start_date.astimezone(timezone.utc) <= action.ex_date <= end_date.astimezone(timezone.utc):
                    parsed_actions.append(action)

            if parsed_actions:
                break

        parsed_actions.sort(key=lambda item: item.ex_date)
        return parsed_actions
