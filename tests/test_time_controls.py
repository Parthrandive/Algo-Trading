from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.utils.time_sync import is_clock_synced, validate_utc_ist_consistency
from src.utils.validation import StreamMonotonicityChecker


def test_is_clock_synced_fails_closed_when_drift_unavailable(monkeypatch):
    monkeypatch.setattr("src.utils.time_sync.get_clock_drift", lambda server="pool.ntp.org": None)
    assert is_clock_synced(threshold_seconds=0.5) is False
    assert is_clock_synced(threshold_seconds=0.5, fail_open=True) is True


def test_validate_utc_ist_consistency():
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now.astimezone(ZoneInfo("Asia/Kolkata"))
    inconsistent_ist = (utc_now + timedelta(minutes=3)).astimezone(ZoneInfo("Asia/Kolkata"))

    assert validate_utc_ist_consistency(utc_now, ist_now) is True
    assert validate_utc_ist_consistency(utc_now, inconsistent_ist) is False


def test_monotonicity_checker_is_scoped_by_symbol_and_interval():
    checker = StreamMonotonicityChecker()
    t1 = datetime(2026, 2, 19, 9, 15, tzinfo=timezone.utc)
    t2 = t1 + timedelta(hours=1)

    assert checker.check("RELIANCE", t1, interval="1h") is True
    assert checker.check("RELIANCE", t1, interval="1d") is True
    assert checker.check("RELIANCE", t2, interval="1h") is True
    assert checker.check("RELIANCE", t1, interval="1h") is False
