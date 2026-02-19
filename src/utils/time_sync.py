import warnings
import ntplib
from time import ctime
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

def get_clock_drift(server="pool.ntp.org") -> float:
    """
    Checks the system clock drift against an NTP server.
    Returns the offset in seconds (positive means system is ahead).
    """
    try:
        client = ntplib.NTPClient()
        response = client.request(server, version=3)
        return response.offset
    except Exception as e:
        logger.warning(f"Failed to check clock drift against {server}: {e}")
        return 0.0

def is_clock_synced(threshold_seconds: float = 1.0, server="pool.ntp.org") -> bool:
    """
    Returns True if clock drift is within threshold.
    """
    drift = get_clock_drift(server)
    is_synced = abs(drift) <= threshold_seconds
    if not is_synced:
        logger.error(f"System clock is out of sync by {drift:.4f} seconds (Threshold: {threshold_seconds}s)")
    return is_synced

def validate_utc_ist_timestamp(dt: datetime) -> bool:
    """
    Validates if a timestamp is timezone-aware and reasonable.
    Reasonable means:
    1. Not in the future (beyond a small buffer).
    2. Not too far in the past (e.g. > 5 years, adjust as needed).
    """
    if dt.tzinfo is None:
        logger.error(f"Timestamp {dt} is naive (missing timezone).")
        return False
    
    now_utc = datetime.now(timezone.utc)
    
    # Check if future (allow 5 min buffer for clock skew/network latency)
    if dt > now_utc + timedelta(minutes=5):
        logger.error(f"Timestamp {dt} is significantly in the future.")
        return False
        
    # Check if too old (e.g. 10 years) - just a sanity check against corrupt years
    if dt < now_utc - timedelta(days=365*10):
        logger.error(f"Timestamp {dt} is unreasonably old.")
        return False

    return True
