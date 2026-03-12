import numpy as np
import pandas as pd
import pytest

from src.agents.technical.models.garch_var import GarchVaRModel


@pytest.fixture
def sample_ohlcv_data():
    """Generate a deterministic heteroskedastic OHLCV dataset."""
    rng = np.random.default_rng(42)
    periods = 260

    omega = 0.000005
    alpha = 0.08
    beta = 0.90

    returns = np.zeros(periods)
    conditional_variance = np.zeros(periods)
    conditional_variance[0] = omega / (1.0 - alpha - beta)

    for idx in range(1, periods):
        shock = rng.normal()
        returns[idx] = np.sqrt(conditional_variance[idx - 1]) * shock
        conditional_variance[idx] = (
            omega
            + alpha * returns[idx] ** 2
            + beta * conditional_variance[idx - 1]
        )

    close = 100.0 * np.exp(np.cumsum(returns))
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]

    intraday_spread = np.abs(returns) * close + 0.25
    high = np.maximum(open_prices, close) + intraday_spread
    low = np.minimum(open_prices, close) - intraday_spread
    volume = 1_000_000 + rng.integers(0, 50_000, size=periods)

    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * periods,
            "timestamp": pd.date_range("2025-01-01", periods=periods, freq="h"),
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_garch_fit_and_forecast(sample_ohlcv_data):
    """Model fits and produces negative VaR with ES beyond VaR."""
    model = GarchVaRModel(window_size=120)

    model.fit(sample_ohlcv_data)
    forecast = model.forecast_risk()

    assert model.is_trained is True
    assert model.fit_result is not None
    assert model.last_convergence_flag == 0
    assert forecast["volatility_forecast"] > 0.0

    assert forecast["parametric_var_95"] < 0.0
    assert forecast["parametric_var_99"] < 0.0
    assert forecast["historical_var_95"] < 0.0
    assert forecast["historical_var_99"] < 0.0

    assert forecast["parametric_es_95"] <= forecast["parametric_var_95"]
    assert forecast["parametric_es_99"] <= forecast["parametric_var_99"]
    assert forecast["historical_es_95"] <= forecast["historical_var_95"]
    assert forecast["historical_es_99"] <= forecast["historical_var_99"]


def test_garch_rolling_forecast_contains_breach_flags(sample_ohlcv_data):
    """Rolling forecast returns the expected columns for backtests."""
    model = GarchVaRModel(window_size=120)

    forecasts = model.rolling_forecast(sample_ohlcv_data, confidence_levels=(0.95,))

    assert not forecasts.empty
    assert "volatility_forecast" in forecasts.columns
    assert "parametric_var_95" in forecasts.columns
    assert "historical_var_95" in forecasts.columns
    assert "breach_95_parametric" in forecasts.columns
    assert "breach_95_historical" in forecasts.columns
    assert forecasts["volatility_forecast"].gt(0.0).all()
    assert forecasts["convergence_flag"].isin({0, -1}).all()

    skipped_ratio = (forecasts["convergence_flag"] == -1).mean()
    assert skipped_ratio <= 0.10

    successful = forecasts["convergence_flag"] == 0
    assert successful.any()
    assert forecasts.loc[successful, "fit_error"].eq("").all()
    assert forecasts.loc[successful, "breach_95_parametric"].notna().all()
    assert forecasts.loc[successful, "breach_95_historical"].notna().all()


def test_garch_backtest_breach_rate_within_expected_bounds(sample_ohlcv_data):
    """Parametric VaR breach frequency stays within reasonable synthetic bounds."""
    model = GarchVaRModel(window_size=120)

    summary = model.backtest_var(
        sample_ohlcv_data,
        confidence_levels=(0.95, 0.99),
        method="parametric",
    )

    stats_95 = summary[95]
    stats_99 = summary[99]

    assert 0.0 <= stats_95["breach_rate"] <= 0.12
    assert abs(stats_95["breach_rate"] - stats_95["expected_breach_rate"]) <= 0.05
    assert 0.0 <= stats_95["p_value"] <= 1.0
    assert stats_95["total_windows"] >= stats_95["observations"]
    assert stats_95["skipped_windows"] <= max(2, int(0.1 * stats_95["total_windows"]))

    assert 0.0 <= stats_99["breach_rate"] <= 0.05
    assert abs(stats_99["breach_rate"] - stats_99["expected_breach_rate"]) <= 0.03
    assert 0.0 <= stats_99["p_value"] <= 1.0
    assert stats_99["total_windows"] >= stats_99["observations"]
    assert stats_99["skipped_windows"] <= max(2, int(0.1 * stats_99["total_windows"]))


def test_garch_historical_backtest_outputs_valid_stats(sample_ohlcv_data):
    """Historical VaR backtest should produce valid bounded statistics."""
    model = GarchVaRModel(window_size=120)

    summary = model.backtest_var(
        sample_ohlcv_data,
        confidence_levels=(0.95, 0.99),
        method="historical",
    )

    for stats in summary.values():
        assert 0.0 <= stats["breach_rate"] <= 1.0
        assert 0.0 <= stats["p_value"] <= 1.0
        assert stats["total_windows"] >= stats["observations"]
        assert stats["skipped_windows"] <= max(2, int(0.1 * stats["total_windows"]))


def test_garch_uses_selected_distribution_for_parametric_tail(sample_ohlcv_data):
    """Parametric VaR/ES should use the fitted distribution, not hard-coded normal tails."""
    model = GarchVaRModel(window_size=120, dist="t")
    model.fit(sample_ohlcv_data)
    forecast = model.forecast_risk(confidence_levels=(0.99,))

    alpha = 0.01
    fit_result = model.fit_result
    assert fit_result is not None

    distribution = fit_result.model.distribution
    dist_params = np.array(
        [fit_result.params[name] for name in distribution.parameter_names()],
        dtype=float,
    )
    quantile = float(np.asarray(distribution.ppf(alpha, dist_params)).reshape(-1)[0])
    partial_moment = None
    partial_moment_calls = (
        lambda: distribution.partial_moment(1, quantile, dist_params),
        lambda: distribution.partial_moment(1, np.array([quantile]), dist_params),
        lambda: distribution.partial_moment(1, quantile, parameters=dist_params),
        lambda: distribution.partial_moment(1, np.array([quantile]), parameters=dist_params),
    )
    for partial_call in partial_moment_calls:
        try:
            partial_moment = float(np.asarray(partial_call()).reshape(-1)[0])
            break
        except Exception:
            continue

    assert partial_moment is not None
    es_standardized = partial_moment / alpha

    mean_forecast = float(fit_result.params.get("mu", 0.0)) / model.scale_factor
    sigma_forecast = forecast["volatility_forecast"]

    expected_var_99 = mean_forecast + sigma_forecast * quantile
    expected_es_99 = mean_forecast + sigma_forecast * es_standardized

    assert forecast["parametric_var_99"] == pytest.approx(expected_var_99, rel=1e-6, abs=1e-9)
    assert forecast["parametric_es_99"] == pytest.approx(expected_es_99, rel=1e-6, abs=1e-9)
