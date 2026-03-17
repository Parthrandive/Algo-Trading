from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.agents.regime.schemas import RegimeState


@dataclass(frozen=True)
class HMMRegimePrediction:
    regime_state: RegimeState
    transition_probability: float
    confidence: float
    hidden_state: int


class HMMRegimeModel:
    """
    NumPy-only HMM-style baseline.

    - Hidden-state estimation: deterministic KMeans-style clustering.
    - Emissions: distance-based soft probabilities to centroids.
    - Transition model: Markov matrix from decoded state sequence.
    """

    def __init__(
        self,
        n_components: int = 4,
        random_state: int = 42,
        feature_columns: Iterable[str] | None = None,
        max_iter: int = 60,
    ) -> None:
        self.n_components = int(max(2, n_components))
        self.random_state = random_state
        self.max_iter = int(max(5, max_iter))
        self.feature_columns = list(
            feature_columns
            or ["close_log_return_zscore", "rolling_vol_20", "macro_directional_flag", "macro_regime_index"]
        )

        self._fitted = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._component_to_regime: dict[int, RegimeState] = {}
        self._transition_matrix: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> None:
        x = self._feature_matrix(df)
        if len(x) < self.n_components + 5:
            raise ValueError("Not enough rows to fit HMMRegimeModel.")

        self._mean = np.mean(x, axis=0)
        self._std = np.std(x, axis=0)
        self._std = np.where(self._std <= 1e-8, 1.0, self._std)
        x_scaled = (x - self._mean) / self._std

        centroids, labels = self._kmeans(x_scaled)
        self._centroids = centroids
        self._component_to_regime = self._map_components_to_regimes()
        self._transition_matrix = self._build_transition_matrix(labels, self.n_components)
        self._fitted = True

    def predict(self, df: pd.DataFrame) -> HMMRegimePrediction:
        if not self._fitted:
            self.fit(df)

        probs, hidden_seq = self._posterior(df)
        hidden_state = int(hidden_seq[-1])
        confidence = float(np.clip(np.max(probs[-1]), 0.0, 1.0))
        regime_state = self._component_to_regime.get(hidden_state, RegimeState.SIDEWAYS)
        transition_probability = self._transition_probability(hidden_state)

        return HMMRegimePrediction(
            regime_state=regime_state,
            transition_probability=transition_probability,
            confidence=round(confidence, 4),
            hidden_state=hidden_state,
        )

    def predict_sequence(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            self.fit(df)
        _, hidden = self._posterior(df)
        return hidden

    def decode_regimes(self, df: pd.DataFrame) -> list[RegimeState]:
        hidden = self.predict_sequence(df)
        return [self._component_to_regime.get(int(state), RegimeState.SIDEWAYS) for state in hidden]

    def _posterior(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self._feature_matrix(df)
        x_scaled = self._transform(x)
        distances = self._pairwise_distances(x_scaled, self._centroids)
        # Soft assignment: exp(-distance) normalized.
        logits = np.exp(-np.clip(distances, 0.0, 40.0))
        probs = logits / np.clip(logits.sum(axis=1, keepdims=True), 1e-12, None)
        hidden = np.argmax(probs, axis=1)
        return probs, hidden

    def _transform(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        assert self._std is not None
        return (x - self._mean) / self._std

    def _transition_probability(self, hidden_state: int) -> float:
        if self._transition_matrix is None:
            return 0.0
        stay_prob = float(np.clip(self._transition_matrix[hidden_state, hidden_state], 0.0, 1.0))
        return round(1.0 - stay_prob, 4)

    @staticmethod
    def _build_transition_matrix(states: np.ndarray, n_states: int, alpha: float = 1.0) -> np.ndarray:
        counts = np.full((n_states, n_states), alpha, dtype=float)
        if len(states) > 1:
            for current, nxt in zip(states[:-1], states[1:]):
                counts[int(current), int(nxt)] += 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        return counts / row_sums

    def _map_components_to_regimes(self) -> dict[int, RegimeState]:
        assert self._centroids is not None
        # Map in original scale for interpretable thresholding.
        means_unscaled = self._centroids * self._std + self._mean
        mapping: dict[int, RegimeState] = {}
        ret_idx = self.feature_columns.index("close_log_return_zscore")
        vol_idx = self.feature_columns.index("rolling_vol_20")
        macro_idx = (
            self.feature_columns.index("macro_regime_index")
            if "macro_regime_index" in self.feature_columns
            else self.feature_columns.index("macro_directional_flag")
        )

        for idx, center in enumerate(means_unscaled):
            ret = float(center[ret_idx])
            vol = float(center[vol_idx])
            macro = float(center[macro_idx])

            if abs(ret) >= 3.0 or vol >= 0.03:
                mapping[idx] = RegimeState.CRISIS
            elif ret >= 0.8:
                mapping[idx] = RegimeState.BULL
            elif ret <= -0.8:
                mapping[idx] = RegimeState.BEAR
            elif abs(macro) >= 0.5:
                mapping[idx] = RegimeState.RBI_BAND_TRANSITION
            else:
                mapping[idx] = RegimeState.SIDEWAYS
        return mapping

    def _kmeans(self, x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        n_rows = len(x_scaled)
        if n_rows < self.n_components:
            raise ValueError("Not enough rows for requested components.")

        # Deterministic/robust initialization from evenly spaced quantiles.
        quantile_idx = np.linspace(0, n_rows - 1, self.n_components, dtype=int)
        sorted_idx = np.argsort(x_scaled[:, 0])
        init_idx = sorted_idx[quantile_idx]
        centroids = x_scaled[init_idx].copy()
        centroids += rng.normal(0.0, 1e-4, size=centroids.shape)

        labels = np.zeros(n_rows, dtype=int)
        for _ in range(self.max_iter):
            distances = self._pairwise_distances(x_scaled, centroids)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for k in range(self.n_components):
                members = x_scaled[labels == k]
                if len(members) == 0:
                    centroids[k] = x_scaled[rng.integers(0, n_rows)]
                else:
                    centroids[k] = members.mean(axis=0)

        return centroids, labels

    @staticmethod
    def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Euclidean distances between each row in a and each centroid in b.
        # Shape: (len(a), len(b))
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        frame = df.copy()
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = 0.0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        matrix = frame[self.feature_columns].to_numpy(dtype=float)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
