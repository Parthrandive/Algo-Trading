import json
from pathlib import Path


def _load_runtime_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "macro_monitor_runtime_v1.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_macro_runtime_config_contract_freeze_day1_cp1():
    config = _load_runtime_config()

    assert config["version"] == "macro-monitor-runtime-v1"
    assert config["schema_version"] == "1.1"

    expected_enum_values = {
        "CPI",
        "WPI",
        "IIP",
        "GDP",
        "INR_USD",
        "BRENT_CRUDE",
        "US_10Y",
        "INDIA_10Y",
        "FII_FLOW",
        "DII_FLOW",
        "REPO_RATE",
        "FX_RESERVES",
        "INDIA_US_10Y_SPREAD",
        "RBI_BULLETIN",
    }
    assert set(config["accepted_indicator_enum_values"]) == expected_enum_values

    expected_day1_catalog = {
        "CPI": {"period": "Monthly", "freshness_window_hours": 48, "unit": "%"},
        "WPI": {"period": "Monthly", "freshness_window_hours": 48, "unit": "%"},
        "IIP": {"period": "Monthly", "freshness_window_hours": 48, "unit": "%"},
        "FII_FLOW": {"period": "Daily", "freshness_window_hours": 4, "unit": "INR_Cr"},
        "DII_FLOW": {"period": "Daily", "freshness_window_hours": 4, "unit": "INR_Cr"},
        "FX_RESERVES": {"period": "Weekly", "freshness_window_hours": 24, "unit": "USD_Bn"},
        "RBI_BULLETIN": {"period": "Irregular", "freshness_window_hours": 24, "unit": "count"},
        "INDIA_US_10Y_SPREAD": {"period": "Daily", "freshness_window_hours": 6, "unit": "bps"},
    }

    indicator_configs = config["indicator_configs"]
    assert set(indicator_configs.keys()) == set(expected_day1_catalog.keys())

    for indicator_name, expected in expected_day1_catalog.items():
        indicator_config = indicator_configs[indicator_name]
        assert indicator_config["period"] == expected["period"]
        assert indicator_config["freshness_window_hours"] == expected["freshness_window_hours"]
        assert indicator_config["unit"] == expected["unit"]

        assert indicator_config["sources"], f"{indicator_name} sources list cannot be empty."
        for source in indicator_config["sources"]:
            assert source["url_pattern"]
            assert source["retry"]["max_attempts"] >= 1
            assert source["retry"]["base_backoff_seconds"] >= 1
            assert source["rate_limit"]["calls"] >= 1
            assert source["rate_limit"]["period_seconds"] >= 1


def test_macro_runtime_config_week3_publish_set():
    config = _load_runtime_config()
    assert set(config["week3_required_publish_set"]) == {
        "CPI",
        "WPI",
        "IIP",
        "FII_FLOW",
        "DII_FLOW",
        "FX_RESERVES",
        "RBI_BULLETIN",
        "INDIA_US_10Y_SPREAD",
    }
