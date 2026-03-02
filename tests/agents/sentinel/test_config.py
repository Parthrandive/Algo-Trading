from datetime import datetime
from zoneinfo import ZoneInfo

from src.agents.sentinel.config import load_default_sentinel_config


def test_default_runtime_config_loads():
    config = load_default_sentinel_config()
    assert config.version == "nse-sentinel-runtime-v1"
    assert config.symbol_universe.version == "week2-core-v1"
    assert "RELIANCE.NS" in config.symbol_universe.all_symbols
    assert len(config.active_sources()) >= 2
    assert config.failover.failure_threshold == 2
    assert config.failover.cooldown_seconds == 60
    assert config.failover.recovery_success_threshold == 2


def test_session_rules_enforce_calendar():
    config = load_default_sentinel_config()
    ist = ZoneInfo("Asia/Kolkata")

    in_session_dt = datetime(2026, 2, 19, 10, 0, tzinfo=ist)
    holiday_dt = datetime(2026, 1, 26, 10, 0, tzinfo=ist)
    out_of_session_dt = datetime(2026, 2, 19, 8, 30, tzinfo=ist)

    assert config.session_rules.is_trading_session(in_session_dt) is True
    assert config.session_rules.is_trading_session(holiday_dt) is False
    assert config.session_rules.is_trading_session(out_of_session_dt) is False
