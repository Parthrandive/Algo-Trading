from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.regime.regime_agent import RegimeAgent
from src.agents.regime.models.hmm_regime import HMMRegimePrediction
from src.agents.regime.models.ood_detector import OODResult
from src.agents.regime.models.pearl_meta import PearlPrediction
from src.agents.regime.schemas import RegimeState, RiskLevel


def _frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(314)
    ret_z = rng.normal(0.6, 0.5, size=rows)
    vol = rng.normal(0.011, 0.002, size=rows)
    macro = rng.choice([-1.0, 0.0, 1.0], size=rows, p=[0.15, 0.7, 0.15])
    close_log_return = ret_z * 0.004
    close = 100 * np.exp(np.cumsum(close_log_return))
    ts = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "close": close,
            "close_log_return": close_log_return,
            "close_log_return_zscore": ret_z,
            "rolling_vol_20": vol,
            "macro_directional_flag": macro,
        }
    )


class _Loader:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def load_features(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        _ = symbol
        return self.df.tail(limit).copy()


class _FakeHMM:
    fitted = True

    def fit(self, df: pd.DataFrame) -> None:
        _ = df

    def predict(self, df: pd.DataFrame) -> HMMRegimePrediction:
        _ = df
        return HMMRegimePrediction(
            regime_state=RegimeState.BULL,
            transition_probability=0.22,
            confidence=0.8,
            hidden_state=1,
        )


class _FakePearl:
    fitted = True

    def fit(self, df: pd.DataFrame) -> None:
        _ = df

    def predict(self, df: pd.DataFrame) -> PearlPrediction:
        _ = df
        return PearlPrediction(
            regime_state=RegimeState.BULL,
            transition_probability=0.18,
            confidence=0.75,
            probabilities={
                RegimeState.BULL.value: 0.75,
                RegimeState.BEAR.value: 0.05,
                RegimeState.SIDEWAYS.value: 0.1,
                RegimeState.CRISIS.value: 0.05,
                RegimeState.RBI_BAND_TRANSITION.value: 0.05,
            },
        )


class _FakeOODAlien:
    fitted = True

    def fit(self, df: pd.DataFrame) -> None:
        _ = df

    def detect(self, recent_df: pd.DataFrame) -> OODResult:
        _ = recent_df
        return OODResult(
            is_warning=True,
            is_alien=True,
            mahalanobis_distance=9.5,
            kl_divergence=8.2,
            risk_level=RiskLevel.NEUTRAL_CASH,
        )


def test_day6_unified_detect_regime_contract():
    agent = RegimeAgent(loader=_Loader(_frame()))
    pred = agent.detect_regime("RELIANCE.NS")

    assert pred.regime_state in {
        RegimeState.BULL,
        RegimeState.BEAR,
        RegimeState.SIDEWAYS,
        RegimeState.CRISIS,
        RegimeState.RBI_BAND_TRANSITION,
        RegimeState.ALIEN,
    }
    assert 0.0 <= pred.transition_probability <= 1.0
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.risk_level in {RiskLevel.FULL_RISK, RiskLevel.REDUCED_RISK, RiskLevel.NEUTRAL_CASH}
    assert pred.model_id == "ensemble_hmm_pearl_ood_v1.0"
    assert pred.details is not None
    assert "hmm" in pred.details
    assert "pearl" in pred.details
    assert "ood" in pred.details


def test_day6_ood_alien_override():
    agent = RegimeAgent(
        loader=_Loader(_frame()),
        hmm_model=_FakeHMM(),
        pearl_model=_FakePearl(),
        ood_detector=_FakeOODAlien(),
    )
    pred = agent.detect_regime("RELIANCE.NS")

    assert pred.regime_state == RegimeState.ALIEN
    assert pred.risk_level == RiskLevel.NEUTRAL_CASH
    assert pred.transition_probability >= 0.7


def test_day6_detect_regime_survives_persistence_failure():
    class _FailingRecorder:
        def save_regime_prediction(self, *args, **kwargs):
            raise RuntimeError("db down")

    agent = RegimeAgent(loader=_Loader(_frame()))
    agent.phase2_recorder = _FailingRecorder()

    pred = agent.detect_regime("RELIANCE.NS")

    assert pred is not None
    assert pred.symbol == "RELIANCE.NS"
