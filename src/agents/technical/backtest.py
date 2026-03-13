from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.agents.technical.features import engineer_features
from src.agents.technical.models.arima_lstm import ArimaLstmHybrid
from src.agents.technical.models.cnn_pattern import CnnPatternClassifier
from src.agents.technical.models.garch_var import GarchVaRModel


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 6
    test_months: int = 1
    step_months: int = 1
    train_days: Optional[int] = None
    test_days: Optional[int] = None
    step_days: Optional[int] = None
    start_date: str = "2019-01-01"


@dataclass(frozen=True)
class WalkForwardSplit:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class TechnicalBacktester:
    """
    Day 5 walk-forward backtesting framework for Technical Agent model families.
    """

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        periods_per_year: float = 252.0,
        arima_epochs: int = 2,
        cnn_epochs: int = 2,
        garch_window: int = 252,
    ):
        self.config = config or WalkForwardConfig()
        self.periods_per_year = periods_per_year
        self.arima_epochs = arima_epochs
        self.cnn_epochs = cnn_epochs
        self.garch_window = garch_window

    def _prepare_market_df(self, market_df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(market_df.columns)
        if missing:
            raise ValueError(f"Missing required columns for backtest: {sorted(missing)}")

        prepared = market_df.copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        # Ensure timestamps are naive for comparison with config strings
        if prepared["timestamp"].dt.tz is not None:
            prepared["timestamp"] = prepared["timestamp"].dt.tz_localize(None)
            
        prepared = prepared.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        prepared = prepared.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        if prepared.empty:
            raise ValueError("No valid rows after timestamp/close cleanup.")
        return prepared

    def _infer_periods_per_year(self, timestamps: pd.Series) -> float:
        if len(timestamps) < 3:
            return self.periods_per_year

        diffs = timestamps.sort_values().diff().dropna()
        if diffs.empty:
            return self.periods_per_year

        median_hours = diffs.median() / pd.Timedelta(hours=1)
        if median_hours <= 1.5:
            return 252.0 * 6.0
        if median_hours <= 36:
            return 252.0
        return 52.0

    def generate_walk_forward_splits(self, market_df: pd.DataFrame) -> List[WalkForwardSplit]:
        prepared = self._prepare_market_df(market_df)

        min_ts = prepared["timestamp"].min()
        max_ts = prepared["timestamp"].max()
        cursor = max(pd.Timestamp(self.config.start_date), min_ts)
        
        # Unit multipliers
        month = pd.DateOffset(months=1)
        day = pd.DateOffset(days=1)

        splits: List[WalkForwardSplit] = []
        split_id = 1
        while True:
            train_start = cursor
            
            # Decide window size (days take precedence if set)
            if self.config.train_days is not None:
                train_end = train_start + (day * self.config.train_days)
            else:
                train_end = train_start + (month * self.config.train_months)
                
            test_start = train_end
            
            if self.config.test_days is not None:
                test_end = test_start + (day * self.config.test_days)
            else:
                test_end = test_start + (month * self.config.test_months)

            if test_end > max_ts:
                break

            train_df = prepared[
                (prepared["timestamp"] >= train_start) & (prepared["timestamp"] < train_end)
            ].copy()
            test_df = prepared[
                (prepared["timestamp"] >= test_start) & (prepared["timestamp"] < test_end)
            ].copy()

            if not train_df.empty and len(test_df) >= 2:
                splits.append(
                    WalkForwardSplit(
                        split_id=split_id,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        train_df=train_df,
                        test_df=test_df,
                    )
                )
                split_id += 1

            if self.config.step_days is not None:
                cursor = cursor + (day * self.config.step_days)
            else:
                cursor = cursor + (month * self.config.step_months)
                
            if cursor >= max_ts:
                break

        return splits

    @staticmethod
    def _max_drawdown(strategy_returns: pd.Series) -> float:
        if strategy_returns.empty:
            return float("nan")
        equity = (1.0 + strategy_returns).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        return float(drawdown.min())

    @staticmethod
    def _directional_accuracy(predicted: pd.Series, actual: pd.Series) -> float:
        aligned = pd.DataFrame({"pred": predicted, "actual": actual}).dropna()
        if aligned.empty:
            return float("nan")

        # Keep only decisions where the model took a direction (non-neutral).
        non_neutral = aligned["pred"] != 0
        if not non_neutral.any():
            return float("nan")
        return float((aligned.loc[non_neutral, "pred"] == aligned.loc[non_neutral, "actual"]).mean())

    def _compute_metrics(self, predictions: pd.DataFrame) -> Dict[str, float]:
        if predictions.empty:
            return {
                "sharpe_ratio": float("nan"),
                "sortino_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "win_rate": float("nan"),
                "profit_factor": float("nan"),
                "directional_accuracy": float("nan"),
                "mean_return": float("nan"),
                "volatility": float("nan"),
                "num_predictions": 0,
            }

        strategy_returns = predictions["strategy_return"].astype(float).dropna()
        if strategy_returns.empty:
            return {
                "sharpe_ratio": float("nan"),
                "sortino_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "win_rate": float("nan"),
                "profit_factor": float("nan"),
                "directional_accuracy": float("nan"),
                "mean_return": float("nan"),
                "volatility": float("nan"),
                "num_predictions": int(len(predictions)),
            }

        periods = self._infer_periods_per_year(predictions["timestamp"])
        mean_return = float(strategy_returns.mean())
        volatility = float(strategy_returns.std(ddof=0))

        sharpe = float("nan")
        if volatility > 0:
            sharpe = float((mean_return / volatility) * np.sqrt(periods))

        downside = strategy_returns[strategy_returns < 0]
        sortino = float("nan")
        if not downside.empty:
            downside_std = float(downside.std(ddof=0))
            if downside_std > 0:
                sortino = float((mean_return / downside_std) * np.sqrt(periods))

        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": self._max_drawdown(strategy_returns),
            "win_rate": float((strategy_returns > 0).mean()),
            "profit_factor": float(profit_factor),
            "directional_accuracy": self._directional_accuracy(
                predictions["predicted_direction"],
                predictions["actual_direction"],
            ),
            "mean_return": mean_return,
            "volatility": volatility,
            "num_predictions": int(len(strategy_returns)),
        }

    @staticmethod
    def _apply_arima_ablation(
        feature_df: pd.DataFrame,
        group: Optional[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        if not group or group == "baseline":
            return feature_df, []

        lowered = group.lower()
        if lowered == "volume":
            to_drop = [col for col in feature_df.columns if "volume" in col.lower()]
        elif lowered == "rsi":
            to_drop = [col for col in feature_df.columns if col.lower() == "rsi"]
        elif lowered == "macd":
            to_drop = [col for col in feature_df.columns if col.lower().startswith("macd")]
        elif lowered == "macro":
            to_drop = [col for col in feature_df.columns if col.lower().startswith("macro_")]
        else:
            raise ValueError(f"Unknown ablation group: {group}")

        protected = {"timestamp", "symbol", "close"}
        safe_drop = [col for col in to_drop if col not in protected]
        return feature_df.drop(columns=safe_drop, errors="ignore"), sorted(safe_drop)

    def _run_arima_lstm_split(
        self,
        split: WalkForwardSplit,
        ablation_group: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        train_features = engineer_features(split.train_df).dropna().reset_index(drop=True)
        full_features = engineer_features(
            pd.concat([split.train_df, split.test_df], ignore_index=True)
        ).dropna().reset_index(drop=True)

        train_features, dropped = self._apply_arima_ablation(train_features, ablation_group)
        full_features, _ = self._apply_arima_ablation(full_features, ablation_group)

        if len(train_features) < 40 or len(full_features) < 60:
            return pd.DataFrame(), dropped

        model = ArimaLstmHybrid(
            arima_order=(1, 1, 0),
            lstm_hidden_size=16,
            lstm_layers=1,
            learning_rate=0.01,
            window_size=5,
        )
        model.fit(train_features, target_col="close", epochs=self.arima_epochs, batch_size=16)

        records: List[Dict[str, float]] = []
        for idx in range(len(full_features)):
            row = full_features.iloc[idx]
            ts = row["timestamp"]
            if ts < split.test_start or ts >= split.test_end:
                continue

            context = full_features.iloc[:idx]
            if len(context) < model.window_size + 5:
                continue

            prev_close = float(context["close"].iloc[-1])
            actual_close = float(row["close"])
            if prev_close <= 0:
                continue

            try:
                forecast_close = float(model.predict(context, target_col="close"))
            except Exception:
                continue

            predicted_return = (forecast_close - prev_close) / prev_close
            realized_return = (actual_close - prev_close) / prev_close
            signal = int(np.sign(predicted_return))
            actual_direction = int(np.sign(realized_return))

            records.append(
                {
                    "timestamp": ts,
                    "strategy_return": signal * realized_return,
                    "realized_return": realized_return,
                    "predicted_direction": signal,
                    "actual_direction": actual_direction,
                }
            )

        return pd.DataFrame.from_records(records), dropped

    def _run_cnn_split(self, split: WalkForwardSplit) -> pd.DataFrame:
        if len(split.train_df) < 40:
            return pd.DataFrame()

        classifier = CnnPatternClassifier(window_size=20, learning_rate=0.01)
        classifier.fit(split.train_df, epochs=self.cnn_epochs, batch_size=16)

        combined = pd.concat([split.train_df, split.test_df], ignore_index=True).sort_values(
            "timestamp"
        )
        combined = combined.reset_index(drop=True)

        records: List[Dict[str, float]] = []
        for idx in range(len(combined)):
            row = combined.iloc[idx]
            ts = row["timestamp"]
            if ts < split.test_start or ts >= split.test_end:
                continue

            context = combined.iloc[:idx]
            if len(context) < classifier.window_size:
                continue

            prev_close = float(context["close"].iloc[-1])
            actual_close = float(row["close"])
            if prev_close <= 0:
                continue

            predicted_class, probabilities = classifier.predict(context.tail(classifier.window_size))
            signal = {"up": 1, "down": -1, "neutral": 0}[predicted_class]
            realized_return = (actual_close - prev_close) / prev_close

            records.append(
                {
                    "timestamp": ts,
                    "strategy_return": signal * realized_return,
                    "realized_return": realized_return,
                    "predicted_direction": signal,
                    "actual_direction": int(np.sign(realized_return)),
                    "confidence": float(probabilities.get(predicted_class, 0.0)),
                }
            )

        return pd.DataFrame.from_records(records)

    def _run_garch_split(self, split: WalkForwardSplit) -> pd.DataFrame:
        combined = pd.concat([split.train_df, split.test_df], ignore_index=True).sort_values(
            "timestamp"
        )
        combined = combined.reset_index(drop=True)
        min_context = max(45, min(self.garch_window, 90))

        records: List[Dict[str, float]] = []
        for idx in range(len(combined)):
            row = combined.iloc[idx]
            ts = row["timestamp"]
            if ts < split.test_start or ts >= split.test_end:
                continue

            context = combined.iloc[:idx]
            if len(context) < min_context:
                continue

            prev_close = float(context["close"].iloc[-1])
            actual_close = float(row["close"])
            if prev_close <= 0:
                continue

            window = min(self.garch_window, max(40, len(context) - 1))
            model = GarchVaRModel(window_size=window, max_fit_retries=1)
            try:
                model.fit(context, price_col="close")
                risk = model.forecast_risk(confidence_levels=(0.95, 0.99))
            except Exception:
                continue

            realized_return = (actual_close - prev_close) / prev_close
            trailing_returns = context["close"].pct_change().dropna().tail(20)
            fallback_mu = float(trailing_returns.mean()) if not trailing_returns.empty else 0.0
            expected_mu = float(model.fit_result.params.get("mu", 0.0)) / model.scale_factor
            if expected_mu == 0.0:
                expected_mu = fallback_mu

            signal = int(np.sign(expected_mu))
            volatility = max(float(risk["volatility_forecast"]), 1e-8)
            position_size = min(1.0, 0.01 / volatility)

            records.append(
                {
                    "timestamp": ts,
                    "strategy_return": signal * position_size * realized_return,
                    "realized_return": realized_return,
                    "predicted_direction": signal,
                    "actual_direction": int(np.sign(realized_return)),
                    "volatility_forecast": volatility,
                    "var_95": float(risk["parametric_var_95"]),
                    "var_99": float(risk["parametric_var_99"]),
                    "es_95": float(risk["parametric_es_95"]),
                    "es_99": float(risk["parametric_es_99"]),
                }
            )

        return pd.DataFrame.from_records(records)

    def run_model_backtests(self, market_df: pd.DataFrame) -> Dict[str, object]:
        prepared = self._prepare_market_df(market_df)
        splits = self.generate_walk_forward_splits(prepared)
        if not splits:
            raise ValueError("No walk-forward splits generated. Increase history span or relax config.")

        model_runners = {
            "arima_lstm": lambda split: self._run_arima_lstm_split(split)[0],
            "cnn_pattern": self._run_cnn_split,
            "garch_var": self._run_garch_split,
        }

        total_test_rows = int(sum(len(split.test_df) for split in splits))
        results: Dict[str, object] = {"config": self.config.__dict__.copy(), "models": {}}

        for model_id, runner in model_runners.items():
            split_metrics: List[Dict[str, float]] = []
            prediction_frames: List[pd.DataFrame] = []

            for split in splits:
                split_predictions = runner(split)
                if split_predictions.empty:
                    continue

                split_predictions = split_predictions.copy()
                split_predictions["split_id"] = split.split_id
                prediction_frames.append(split_predictions)

                metrics = self._compute_metrics(split_predictions)
                metrics["split_id"] = split.split_id
                split_metrics.append(metrics)

            if prediction_frames:
                all_predictions = pd.concat(prediction_frames, ignore_index=True)
                aggregate_metrics = self._compute_metrics(all_predictions)
                num_predictions = int(len(all_predictions))
            else:
                all_predictions = pd.DataFrame()
                aggregate_metrics = self._compute_metrics(all_predictions)
                num_predictions = 0

            coverage_ratio = float(num_predictions / total_test_rows) if total_test_rows else 0.0
            results["models"][model_id] = {
                "metrics": aggregate_metrics,
                "num_splits": len(split_metrics),
                "num_predictions": num_predictions,
                "coverage_ratio": coverage_ratio,
                "split_metrics": split_metrics,
            }

        return results

    def run_ablation(
        self,
        market_df: pd.DataFrame,
        groups: Sequence[str] = ("volume", "rsi", "macd", "macro"),
    ) -> Dict[str, object]:
        prepared = self._prepare_market_df(market_df)
        splits = self.generate_walk_forward_splits(prepared)
        if not splits:
            raise ValueError("No walk-forward splits generated. Increase history span or relax config.")

        group_order = ["baseline", *groups]
        group_results: Dict[str, Dict[str, object]] = {}

        for group in group_order:
            ablation_group = None if group == "baseline" else group
            prediction_frames: List[pd.DataFrame] = []
            dropped_union: set[str] = set()

            for split in splits:
                split_predictions, dropped_cols = self._run_arima_lstm_split(
                    split,
                    ablation_group=ablation_group,
                )
                dropped_union.update(dropped_cols)
                if not split_predictions.empty:
                    prediction_frames.append(split_predictions)

            if prediction_frames:
                all_predictions = pd.concat(prediction_frames, ignore_index=True)
                metrics = self._compute_metrics(all_predictions)
                num_predictions = int(len(all_predictions))
            else:
                metrics = self._compute_metrics(pd.DataFrame())
                num_predictions = 0

            group_results[group] = {
                "metrics": metrics,
                "num_predictions": num_predictions,
                "dropped_columns": sorted(dropped_union),
            }

        baseline = group_results["baseline"]["metrics"]
        for group, payload in group_results.items():
            metrics = payload["metrics"]
            payload["delta_vs_baseline"] = {
                "sharpe_delta": float(metrics["sharpe_ratio"] - baseline["sharpe_ratio"])
                if pd.notna(metrics["sharpe_ratio"]) and pd.notna(baseline["sharpe_ratio"])
                else float("nan"),
                "accuracy_delta": float(
                    metrics["directional_accuracy"] - baseline["directional_accuracy"]
                )
                if pd.notna(metrics["directional_accuracy"])
                and pd.notna(baseline["directional_accuracy"])
                else float("nan"),
            }

        return {"config": self.config.__dict__.copy(), "groups": group_results}

    @staticmethod
    def _fmt_metric(value: object) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "N/A"
            if np.isinf(value):
                return "inf"
            return f"{float(value):.4f}"
        return str(value)

    def render_backtest_report(self, backtest_results: Dict[str, object]) -> str:
        generated = datetime.now(timezone.utc).isoformat()
        lines = [
            "# Technical Agent Backtest Report (Day 5)",
            "",
            f"- Generated: `{generated}`",
            f"- Walk-forward config: `{backtest_results.get('config', {})}`",
            "",
            "## Aggregate Metrics",
            "",
            "| Model | Sharpe | Sortino | Max Drawdown | Win Rate | Profit Factor | Directional Accuracy | Coverage | Predictions |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        models = backtest_results.get("models", {})
        for model_id, payload in models.items():
            metrics = payload.get("metrics", {})
            row = [
                model_id,
                self._fmt_metric(metrics.get("sharpe_ratio")),
                self._fmt_metric(metrics.get("sortino_ratio")),
                self._fmt_metric(metrics.get("max_drawdown")),
                self._fmt_metric(metrics.get("win_rate")),
                self._fmt_metric(metrics.get("profit_factor")),
                self._fmt_metric(metrics.get("directional_accuracy")),
                self._fmt_metric(payload.get("coverage_ratio")),
                self._fmt_metric(payload.get("num_predictions")),
            ]
            lines.append("| " + " | ".join(row) + " |")

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Metrics are computed on strategy returns from walk-forward predictions.",
                "- GARCH strategy uses a volatility-scaled directional proxy from fitted mean/trailing drift.",
                "- This report is generated from whichever dataset is passed to `TechnicalBacktester`.",
            ]
        )
        return "\n".join(lines)

    def render_ablation_report(self, ablation_results: Dict[str, object]) -> str:
        generated = datetime.now(timezone.utc).isoformat()
        lines = [
            "# Technical Agent Ablation Report (Day 5)",
            "",
            f"- Generated: `{generated}`",
            f"- Walk-forward config: `{ablation_results.get('config', {})}`",
            "",
            "## ARIMA-LSTM Feature Group Ablation",
            "",
            "| Group | Sharpe | Directional Accuracy | Sharpe Delta vs Baseline | Accuracy Delta vs Baseline | Predictions | Dropped Columns |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]

        groups = ablation_results.get("groups", {})
        for group_name, payload in groups.items():
            metrics = payload.get("metrics", {})
            deltas = payload.get("delta_vs_baseline", {})
            dropped = payload.get("dropped_columns", [])
            dropped_text = ", ".join(dropped) if dropped else "None"

            row = [
                group_name,
                self._fmt_metric(metrics.get("sharpe_ratio")),
                self._fmt_metric(metrics.get("directional_accuracy")),
                self._fmt_metric(deltas.get("sharpe_delta")),
                self._fmt_metric(deltas.get("accuracy_delta")),
                self._fmt_metric(payload.get("num_predictions")),
                dropped_text,
            ]
            lines.append("| " + " | ".join(row) + " |")

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Ablation currently targets ARIMA-LSTM feature engineering groups.",
                "- `macro` group removes columns prefixed with `macro_` when present.",
            ]
        )
        return "\n".join(lines)

    def write_reports(
        self,
        backtest_results: Dict[str, object],
        ablation_results: Dict[str, object],
        backtest_report_path: str = "docs/reports/technical_agent_backtest.md",
        ablation_report_path: str = "docs/reports/technical_agent_ablation.md",
    ) -> Tuple[Path, Path]:
        backtest_path = Path(backtest_report_path)
        ablation_path = Path(ablation_report_path)
        backtest_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_path.parent.mkdir(parents=True, exist_ok=True)

        backtest_path.write_text(self.render_backtest_report(backtest_results), encoding="utf-8")
        ablation_path.write_text(self.render_ablation_report(ablation_results), encoding="utf-8")
        return backtest_path, ablation_path


def run_day5_backtests(
    market_df: pd.DataFrame,
    config: Optional[WalkForwardConfig] = None,
    backtest_report_path: str = "docs/reports/technical_agent_backtest.md",
    ablation_report_path: str = "docs/reports/technical_agent_ablation.md",
) -> Dict[str, object]:
    """
    Convenience entrypoint for Day 5 execution from a prepared OHLCV DataFrame.
    """
    backtester = TechnicalBacktester(config=config)
    backtest_results = backtester.run_model_backtests(market_df)
    ablation_results = backtester.run_ablation(market_df)
    backtester.write_reports(
        backtest_results=backtest_results,
        ablation_results=ablation_results,
        backtest_report_path=backtest_report_path,
        ablation_report_path=ablation_report_path,
    )
    return {"backtest": backtest_results, "ablation": ablation_results}
