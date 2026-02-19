import warnings
from time import ctime
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
    Stateful checker for timestamp monotonicity per symbol.
    """
    def __init__(self):
        self.last_seen: dict[str, datetime] = {}

    def check(self, symbol: str, timestamp: datetime) -> bool:
        """
        Returns True if timestamp is strictly greater than the last seen timestamp for the symbol.
        Updates the last seen timestamp if valid.
        """
        last = self.last_seen.get(symbol)
        
        if last is None:
            self.last_seen[symbol] = timestamp
            return True
            
        if timestamp <= last:
            return False
            
        self.last_seen[symbol] = timestamp
        return True

    def reset(self, symbol: str):
        if symbol in self.last_seen:
            del self.last_seen[symbol]

