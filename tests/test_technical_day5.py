import numpy as np
import pandas as pd
import pytest

from src.agents.technical.backtest import TechnicalBacktester, WalkForwardConfig


@pytest.fixture
def sample_backtest_data():
    """Synthetic OHLCV series spanning multiple months for walk-forward tests."""
    rng = np.random.default_rng(7)
    periods = 210
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")

    drift = 0.0006
    noise = rng.normal(loc=0.0, scale=0.012, size=periods)
    returns = drift + noise
    close = 100.0 * np.exp(np.cumsum(returns))
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]

    spread = np.abs(noise) * close + 0.2
    high = np.maximum(open_prices, close) + spread
    low = np.minimum(open_prices, close) - spread
    volume = 900_000 + rng.integers(0, 100_000, size=periods)

    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * periods,
            "timestamp": timestamps,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def day5_backtester():
    return TechnicalBacktester(
        config=WalkForwardConfig(train_months=2, test_months=1, step_months=2, start_date="2024-01-01"),
        periods_per_year=252.0,
        arima_epochs=1,
        cnn_epochs=1,
        garch_window=40,
    )


def test_walk_forward_splits_generated(sample_backtest_data, day5_backtester):
    splits = day5_backtester.generate_walk_forward_splits(sample_backtest_data)
    assert len(splits) >= 2
    assert all(len(split.train_df) > 0 for split in splits)
    assert all(len(split.test_df) >= 2 for split in splits)


def test_day5_backtest_runs_all_model_families(sample_backtest_data, day5_backtester):
    results = day5_backtester.run_model_backtests(sample_backtest_data)
    models = results["models"]

    assert "arima_lstm" in models
    assert "cnn_pattern" in models
    assert "garch_var" in models

    required_metric_keys = {
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "directional_accuracy",
        "num_predictions",
    }

    for payload in models.values():
        metrics = payload["metrics"]
        assert required_metric_keys.issubset(metrics.keys())
        assert payload["num_splits"] >= 0
        assert payload["num_predictions"] >= 0


def test_day5_ablation_and_report_generation(sample_backtest_data, day5_backtester, tmp_path):
    backtest_results = day5_backtester.run_model_backtests(sample_backtest_data)
    ablation_results = day5_backtester.run_ablation(sample_backtest_data)

    groups = ablation_results["groups"]
    assert "baseline" in groups
    assert "volume" in groups
    assert "rsi" in groups
    assert "macd" in groups
    assert "macro" in groups

    for payload in groups.values():
        assert "metrics" in payload
        assert "delta_vs_baseline" in payload

    backtest_report = day5_backtester.render_backtest_report(backtest_results)
    ablation_report = day5_backtester.render_ablation_report(ablation_results)
    assert "Technical Agent Backtest Report" in backtest_report
    assert "Technical Agent Ablation Report" in ablation_report

    backtest_path = tmp_path / "technical_agent_backtest.md"
    ablation_path = tmp_path / "technical_agent_ablation.md"
    day5_backtester.write_reports(
        backtest_results=backtest_results,
        ablation_results=ablation_results,
        backtest_report_path=str(backtest_path),
        ablation_report_path=str(ablation_path),
    )

    assert backtest_path.exists()
    assert ablation_path.exists()
