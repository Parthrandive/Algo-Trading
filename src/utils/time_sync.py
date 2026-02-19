import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import ntplib

logger = logging.getLogger(__name__)


def get_clock_drift(server: str = "pool.ntp.org") -> Optional[float]:
    """
    Checks the system clock drift against an NTP server.
    Returns the offset in seconds (positive means system is ahead).
    Returns None when drift cannot be determined.
    """
    try:
        client = ntplib.NTPClient()
        response = client.request(server, version=3)
        return response.offset
    except Exception as e:
        logger.error(f"Failed to check clock drift against {server}: {e}")
        return None


def is_clock_synced(threshold_seconds: float = 1.0, server: str = "pool.ntp.org", fail_open: bool = False) -> bool:
    """
    Returns True if clock drift is within threshold.
    If drift is unavailable, returns fail_open (False by default).
    """
    drift = get_clock_drift(server)
    if drift is None:
        if fail_open:
            logger.warning("Clock drift check unavailable; continuing in fail-open mode.")
            return True
        logger.error("Clock drift check unavailable; treating as unsynced.")
        return False

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


def validate_utc_ist_consistency(utc_dt: datetime, ist_dt: datetime, tolerance_seconds: float = 0.5) -> bool:
    """
    Validates that UTC and IST timestamps represent the same instant.
    """
    if utc_dt.tzinfo is None or ist_dt.tzinfo is None:
        logger.error("UTC/IST consistency check failed: one or both timestamps are timezone-naive.")
        return False

    utc_normalized = utc_dt.astimezone(timezone.utc)
    ist_normalized = ist_dt.astimezone(timezone.utc)
    delta_seconds = abs((utc_normalized - ist_normalized).total_seconds())
    if delta_seconds > tolerance_seconds:
        logger.error(
            "UTC/IST consistency check failed: delta=%ss exceeds tolerance=%ss.",
            delta_seconds,
            tolerance_seconds,
        )
        return False
    return True
