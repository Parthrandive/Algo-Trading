from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agents.regime.schemas import RegimeState


@dataclass(frozen=True)
class PearlPrediction:
    regime_state: RegimeState
    transition_probability: float
    confidence: float
    probabilities: dict[str, float]


class PearlMetaModel:
    """
    PEARL-like adaptive model (lightweight approximation).

    This implementation keeps a base probability model and performs rapid context
    adaptation over the latest window to rebalance regime probabilities.
    """

    def __init__(
        self,
        adaptation_window: int = 60,
        max_adaptation_weight: float = 0.7,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.adaptation_window = int(max(10, adaptation_window))
        self.max_adaptation_weight = float(np.clip(max_adaptation_weight, 0.1, 0.95))
        self.feature_columns = feature_columns or [
            "close_log_return_zscore",
            "rolling_vol_20",
            "macro_directional_flag",
            "macro_regime_index",
        ]
        self._states = [
            RegimeState.BULL,
            RegimeState.BEAR,
            RegimeState.SIDEWAYS,
            RegimeState.CRISIS,
            RegimeState.RBI_BAND_TRANSITION,
        ]
        self._fitted = False
        self._base_means: dict[RegimeState, np.ndarray] = {}
        self._global_mean: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> None:
        x, labels = self._prepare_training_data(df)
        self._global_mean = np.mean(x, axis=0)
        self._base_means = {}

        for state in self._states:
            rows = x[np.array([label == state for label in labels])]
            if len(rows) == 0:
                self._base_means[state] = self._global_mean.copy()
            else:
                self._base_means[state] = np.mean(rows, axis=0)
        self._fitted = True

    def predict(self, df: pd.DataFrame) -> PearlPrediction:
        if not self._fitted:
            self.fit(df)
        assert self._global_mean is not None

        x = self._prepare_features(df)
        latest = x[-1]
        context = x[-self.adaptation_window :]

        base_probs = self._base_probabilities(latest)
        context_state = self._heuristic_state(context.mean(axis=0))
        adaptation_weight = min(self.max_adaptation_weight, len(context) / self.adaptation_window * self.max_adaptation_weight)

        adapted_probs = base_probs.copy()
        adapted_probs[context_state] = adapted_probs.get(context_state, 0.0) + adaptation_weight
        adapted_probs = self._normalize(adapted_probs)

        top_state = max(adapted_probs, key=adapted_probs.get)
        top_prob = float(adapted_probs[top_state])
        second_prob = sorted(adapted_probs.values(), reverse=True)[1]
        transition_probability = float(np.clip(1.0 - (top_prob - second_prob), 0.0, 1.0))

        return PearlPrediction(
            regime_state=top_state,
            transition_probability=round(transition_probability, 4),
            confidence=round(top_prob, 4),
            probabilities={state.value: round(prob, 6) for state, prob in adapted_probs.items()},
        )

    def _base_probabilities(self, latest: np.ndarray) -> dict[RegimeState, float]:
        scores: dict[RegimeState, float] = {}
        for state, mean_vec in self._base_means.items():
            dist = np.linalg.norm(latest - mean_vec)
            scores[state] = float(np.exp(-dist))
        return self._normalize(scores)

    @staticmethod
    def _normalize(scores: dict[RegimeState, float]) -> dict[RegimeState, float]:
        total = sum(scores.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(scores))
            return {k: uniform for k in scores}
        return {k: v / total for k, v in scores.items()}

    def _prepare_training_data(self, df: pd.DataFrame) -> tuple[np.ndarray, list[RegimeState]]:
        x = self._prepare_features(df)
        labels = [self._heuristic_state(row) for row in x]
        return x, labels

    def _heuristic_state(self, row: np.ndarray) -> RegimeState:
        idx = {name: i for i, name in enumerate(self.feature_columns)}
        ret_z = float(row[idx.get("close_log_return_zscore", 0)])
        vol = float(row[idx.get("rolling_vol_20", 1 if len(row) > 1 else 0)])
        macro = float(row[idx.get("macro_regime_index", idx.get("macro_directional_flag", 0))])

        if abs(ret_z) >= 3.0 or vol >= 0.03:
            return RegimeState.CRISIS
        if ret_z >= 0.8:
            return RegimeState.BULL
        if ret_z <= -0.8:
            return RegimeState.BEAR
        if abs(macro) >= 0.5:
            return RegimeState.RBI_BAND_TRANSITION
        return RegimeState.SIDEWAYS

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        frame = df.copy()
        if "close_log_return_zscore" not in frame.columns:
            ret = pd.to_numeric(frame.get("close_log_return"), errors="coerce")
            rolling = ret.rolling(window=30, min_periods=5)
            frame["close_log_return_zscore"] = (ret - rolling.mean()) / rolling.std().replace(0, np.nan)

        if "rolling_vol_20" not in frame.columns:
            ret = pd.to_numeric(frame.get("close_log_return"), errors="coerce")
            frame["rolling_vol_20"] = ret.rolling(window=20, min_periods=5).std()

        if "macro_directional_flag" not in frame.columns:
            frame["macro_directional_flag"] = 0.0

        if "macro_regime_index" not in frame.columns:
            frame["macro_regime_index"] = pd.to_numeric(frame.get("macro_directional_flag"), errors="coerce").fillna(0.0)

        cols = self.feature_columns
        for col in cols:
            if col not in frame.columns:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        x = frame[cols].to_numpy(dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
