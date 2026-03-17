from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agents.regime.schemas import RiskLevel


@dataclass(frozen=True)
class OODResult:
    is_warning: bool
    is_alien: bool
    mahalanobis_distance: float
    kl_divergence: float
    risk_level: RiskLevel


class OODDetector:
    """
    OOD and alien-state detector using Mahalanobis + KL divergence.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        warning_mahalanobis: float = 3.0,
        alien_mahalanobis: float = 5.0,
        warning_kl: float = 1.0,
        alien_kl: float = 6.0,
    ) -> None:
        self.feature_columns = feature_columns or [
            "close_log_return_zscore",
            "rolling_vol_20",
            "macro_directional_flag",
            "macro_regime_index",
        ]
        self.warning_mahalanobis = float(warning_mahalanobis)
        self.alien_mahalanobis = float(alien_mahalanobis)
        self.warning_kl = float(warning_kl)
        self.alien_kl = float(alien_kl)
        self._fitted = False
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._ref_stats: dict[str, tuple[float, float]] = {}

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> None:
        x = self._feature_matrix(df)
        if len(x) < 20:
            raise ValueError("Not enough rows to fit OODDetector.")
        self._mean = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0]) * 1e-6
        self._inv_cov = np.linalg.pinv(cov)

        self._ref_stats = {}
        for idx, col in enumerate(self.feature_columns):
            series = x[:, idx]
            self._ref_stats[col] = (float(np.mean(series)), float(np.std(series) + 1e-8))
        self._fitted = True

    def detect(self, recent_df: pd.DataFrame) -> OODResult:
        if not self._fitted:
            self.fit(recent_df)
        assert self._mean is not None
        assert self._inv_cov is not None

        x = self._feature_matrix(recent_df)
        latest = x[-1]
        mahalanobis = self._mahalanobis(latest)
        kl = self._gaussian_kl(latest)

        is_alien = mahalanobis >= self.alien_mahalanobis or kl >= self.alien_kl
        is_warning = is_alien or mahalanobis >= self.warning_mahalanobis or kl >= self.warning_kl

        if is_alien:
            risk = RiskLevel.NEUTRAL_CASH
        elif is_warning:
            risk = RiskLevel.REDUCED_RISK
        else:
            risk = RiskLevel.FULL_RISK

        return OODResult(
            is_warning=is_warning,
            is_alien=is_alien,
            mahalanobis_distance=round(float(mahalanobis), 6),
            kl_divergence=round(float(kl), 6),
            risk_level=risk,
        )

    def _mahalanobis(self, sample: np.ndarray) -> float:
        assert self._mean is not None
        assert self._inv_cov is not None
        delta = sample - self._mean
        dist_sq = float(delta.T @ self._inv_cov @ delta)
        return float(np.sqrt(max(dist_sq, 0.0)))

    def _gaussian_kl(self, sample: np.ndarray) -> float:
        # Point-sample approximation: 0.5 * z^2 averaged across features.
        kl_sum = 0.0
        for idx, col in enumerate(self.feature_columns):
            mu_q = float(sample[idx])
            mu_p, sigma_p = self._ref_stats[col]
            sigma_p = max(sigma_p, 1e-6)
            z = (mu_q - mu_p) / sigma_p
            term = 0.5 * (z**2)
            kl_sum += float(max(term, 0.0))
        return float(kl_sum / len(self.feature_columns))

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
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

        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        x = frame[self.feature_columns].to_numpy(dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
