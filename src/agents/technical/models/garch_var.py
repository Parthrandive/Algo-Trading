import os
import tempfile
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

_MPL_CONFIG_DIR = os.path.join(tempfile.gettempdir(), "matplotlib")
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)

from arch import arch_model


class GarchVaRModel:
    """
    GARCH(1,1) risk model for one-step-ahead volatility, VaR, and ES forecasts.

    The model expects return series in decimal form. When given OHLCV bars it will
    derive simple returns from the configured `price_col`.
    """

    def __init__(
        self,
        window_size: int = 252,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        dist: str = "normal",
        scale_factor: float = 100.0,
    ):
        self.window_size = window_size
        self.p = p
        self.q = q
        self.mean = mean
        self.dist = dist
        self.scale_factor = scale_factor

        self.fit_result = None
        self.training_returns: Optional[pd.Series] = None
        self.is_trained = False
        self.last_convergence_flag: Optional[int] = None

    def _coerce_returns(
        self,
        data: Union[pd.DataFrame, pd.Series],
        price_col: str = "close",
    ) -> pd.Series:
        """Normalize inputs into a clean decimal return series."""
        if isinstance(data, pd.Series):
            returns = data.copy()
        elif isinstance(data, pd.DataFrame):
            frame = data.copy()
            if "timestamp" in frame.columns:
                frame = frame.set_index("timestamp")

            if price_col in frame.columns:
                returns = frame[price_col].pct_change()
            elif "returns" in frame.columns:
                returns = frame["returns"]
            elif "return" in frame.columns:
                returns = frame["return"]
            else:
                raise ValueError(
                    f"Expected `{price_col}`, `returns`, or `return` column in dataframe."
                )
        else:
            raise TypeError("Data must be a pandas DataFrame or Series.")

        returns = pd.to_numeric(returns, errors="coerce").dropna()
        if returns.empty:
            raise ValueError("Return series is empty after dropping NaNs.")
        return returns.astype(float)

    def _fit_series(self, returns: pd.Series):
        if len(returns) < max(30, self.p + self.q + 10):
            raise ValueError("Not enough return observations to fit GARCH.")

        scaled_returns = returns * self.scale_factor
        model = arch_model(
            scaled_returns,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False,
        )
        return model.fit(disp="off", show_warning=False)

    @staticmethod
    def _normal_expected_shortfall(mean: float, sigma: float, tail_probability: float) -> float:
        z_score = norm.ppf(tail_probability)
        return mean - sigma * norm.pdf(z_score) / tail_probability

    def _forecast_from_result(
        self,
        fit_result,
        historical_returns: pd.Series,
        confidence_levels: Sequence[float],
    ) -> Dict[str, float]:
        forecast = fit_result.forecast(horizon=1, reindex=False)
        mean_forecast = float(fit_result.params.get("mu", 0.0)) / self.scale_factor
        sigma_forecast = (
            float(np.sqrt(forecast.variance.values[-1, 0])) / self.scale_factor
        )

        metrics: Dict[str, float] = {"volatility_forecast": sigma_forecast}

        for level in confidence_levels:
            if not 0.0 < level < 1.0:
                raise ValueError("Confidence levels must be between 0 and 1.")

            percentile = int(round(level * 100))
            tail_probability = 1.0 - level
            z_score = norm.ppf(tail_probability)

            parametric_var = mean_forecast + sigma_forecast * z_score
            parametric_es = self._normal_expected_shortfall(
                mean_forecast, sigma_forecast, tail_probability
            )

            historical_var = float(historical_returns.quantile(tail_probability))
            historical_tail = historical_returns[historical_returns <= historical_var]
            historical_es = float(
                historical_tail.mean() if not historical_tail.empty else historical_var
            )

            metrics[f"parametric_var_{percentile}"] = parametric_var
            metrics[f"parametric_es_{percentile}"] = parametric_es
            metrics[f"historical_var_{percentile}"] = historical_var
            metrics[f"historical_es_{percentile}"] = historical_es

        return metrics

    def fit(
        self,
        data: Union[pd.DataFrame, pd.Series],
        price_col: str = "close",
    ) -> "GarchVaRModel":
        """
        Fit the GARCH model on the most recent window of returns.
        """
        returns = self._coerce_returns(data, price_col=price_col)
        training_window = min(len(returns), self.window_size)
        self.training_returns = returns.iloc[-training_window:]
        self.fit_result = self._fit_series(self.training_returns)
        self.last_convergence_flag = int(getattr(self.fit_result, "convergence_flag", 0))
        self.is_trained = True
        return self

    def forecast_risk(
        self,
        confidence_levels: Sequence[float] = (0.95, 0.99),
        historical_window: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Produce 1-step-ahead volatility, VaR, and ES forecasts.
        """
        if not self.is_trained or self.fit_result is None or self.training_returns is None:
            raise RuntimeError("Model is not trained yet.")

        history_window = historical_window or min(
            len(self.training_returns), self.window_size
        )
        historical_returns = self.training_returns.iloc[-history_window:]
        return self._forecast_from_result(
            self.fit_result, historical_returns, confidence_levels
        )

    def rolling_forecast(
        self,
        data: Union[pd.DataFrame, pd.Series],
        price_col: str = "close",
        confidence_levels: Sequence[float] = (0.95, 0.99),
        window_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run rolling 1-step-ahead forecasts across the return series.
        """
        returns = self._coerce_returns(data, price_col=price_col)
        effective_window = window_size or self.window_size

        if len(returns) <= effective_window:
            raise ValueError(
                f"Need more than {effective_window} return observations for rolling forecasts."
            )

        records = []
        for end_idx in range(effective_window, len(returns)):
            window_returns = returns.iloc[end_idx - effective_window : end_idx]
            fit_result = self._fit_series(window_returns)
            forecast_metrics = self._forecast_from_result(
                fit_result, window_returns, confidence_levels
            )

            realized_return = float(returns.iloc[end_idx])
            timestamp = returns.index[end_idx]
            record = {
                "timestamp": timestamp,
                "realized_return": realized_return,
                "convergence_flag": int(getattr(fit_result, "convergence_flag", 0)),
                **forecast_metrics,
            }

            for level in confidence_levels:
                percentile = int(round(level * 100))
                record[f"breach_{percentile}_parametric"] = (
                    realized_return < forecast_metrics[f"parametric_var_{percentile}"]
                )
                record[f"breach_{percentile}_historical"] = (
                    realized_return < forecast_metrics[f"historical_var_{percentile}"]
                )

            records.append(record)

        return pd.DataFrame.from_records(records).set_index("timestamp")

    @staticmethod
    def kupiec_pof_test(
        breaches: Iterable[bool],
        expected_breach_rate: float,
    ) -> Dict[str, float]:
        """
        Kupiec Proportion of Failures test for VaR breach calibration.
        """
        breach_array = np.asarray(list(breaches), dtype=bool)
        if breach_array.size == 0:
            raise ValueError("Kupiec test requires at least one observation.")

        if not 0.0 < expected_breach_rate < 1.0:
            raise ValueError("Expected breach rate must be between 0 and 1.")

        observations = int(breach_array.size)
        breaches_count = int(breach_array.sum())
        observed_rate = breaches_count / observations

        epsilon = 1e-12
        observed_rate_clipped = np.clip(observed_rate, epsilon, 1.0 - epsilon)
        expected_rate_clipped = np.clip(
            expected_breach_rate, epsilon, 1.0 - epsilon
        )

        log_null = (
            breaches_count * np.log(expected_rate_clipped)
            + (observations - breaches_count) * np.log(1.0 - expected_rate_clipped)
        )
        log_alt = (
            breaches_count * np.log(observed_rate_clipped)
            + (observations - breaches_count) * np.log(1.0 - observed_rate_clipped)
        )

        likelihood_ratio = max(0.0, -2.0 * (log_null - log_alt))
        p_value = float(1.0 - chi2.cdf(likelihood_ratio, df=1))

        return {
            "observations": observations,
            "breaches": breaches_count,
            "breach_rate": observed_rate,
            "expected_breach_rate": expected_breach_rate,
            "likelihood_ratio": likelihood_ratio,
            "p_value": p_value,
        }

    def backtest_var(
        self,
        data: Union[pd.DataFrame, pd.Series],
        price_col: str = "close",
        confidence_levels: Sequence[float] = (0.95, 0.99),
        method: str = "parametric",
        window_size: Optional[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Backtest VaR calibration using rolling forecasts and the Kupiec POF test.
        """
        if method not in {"parametric", "historical"}:
            raise ValueError("method must be either `parametric` or `historical`.")

        forecasts = self.rolling_forecast(
            data=data,
            price_col=price_col,
            confidence_levels=confidence_levels,
            window_size=window_size,
        )

        summary: Dict[int, Dict[str, float]] = {}
        for level in confidence_levels:
            percentile = int(round(level * 100))
            breach_column = f"breach_{percentile}_{method}"
            kupiec = self.kupiec_pof_test(
                forecasts[breach_column].tolist(),
                expected_breach_rate=1.0 - level,
            )
            summary[percentile] = kupiec

        return summary
