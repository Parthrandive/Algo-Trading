import warnings
from datetime import datetime
try:
    import ntplib
except ImportError:
    ntplib = None

def check_clock_sync(server="pool.ntp.org") -> float:
    """
    Checks the system clock drift against an NTP server.
    Returns the offset in seconds.
    """
    if ntplib is None:
        warnings.warn("ntplib not installed. Clock sync check skipped (returning 0.0).")
        return 0.0
        
    client = ntplib.NTPClient()
    response = client.request(server, version=3)
    return response.offset

def validate_timestamp_monotonicity(timestamps: list[datetime]) -> bool:
    """
    Checks if a list of timestamps is monotonically increasing.
    """
    return all(x < y for x, y in zip(timestamps, timestamps[1:]))

class StreamMonotonicityChecker:
    """
    Stateful checker for timestamp monotonicity per symbol and interval.
    """
    def __init__(self):
        self.last_seen: dict[tuple[str, str], datetime] = {}

    @staticmethod
    def _key(symbol: str, interval: str | None) -> tuple[str, str]:
        return (symbol, interval or "__default__")

    def check(self, symbol: str, timestamp: datetime, interval: str | None = None) -> bool:
        """
        Returns True if timestamp is strictly greater than the last seen timestamp
        for the symbol+interval stream.
        Updates the last seen timestamp if valid.
        """
        key = self._key(symbol, interval)
        last = self.last_seen.get(key)
        
        if last is None:
            self.last_seen[key] = timestamp
            return True
            
        if timestamp <= last:
            return False
            
        self.last_seen[key] = timestamp
        return True

    def reset(self, symbol: str, interval: str | None = None):
        if interval is None:
            to_delete = [key for key in self.last_seen if key[0] == symbol]
            for key in to_delete:
                del self.last_seen[key]
            return

        key = self._key(symbol, interval)
        if key in self.last_seen:
            del self.last_seen[key]
