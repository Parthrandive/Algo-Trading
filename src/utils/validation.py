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
