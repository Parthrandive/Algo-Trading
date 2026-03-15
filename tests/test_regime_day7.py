from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.regime.regime_agent import RegimeAgent


ROOT = Path(__file__).resolve().parents[1]


class _StaticLoader:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def load_features(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        _ = symbol
        return self.df.tail(limit).copy()


def _stable_frame(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(777)
    ret_z = rng.normal(0.4, 0.25, size=rows)
    vol = rng.normal(0.009, 0.001, size=rows)
    macro = np.zeros(rows)
    close_log_return = ret_z * 0.004
    close = 100 * np.exp(np.cumsum(close_log_return))
    ts = pd.date_range("2024-03-01", periods=rows, freq="D", tz="UTC")
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


def test_day7_model_cards_exist_and_contract():
    cards = [
        ROOT / "data" / "models" / "hmm_regime" / "model_card.json",
        ROOT / "data" / "models" / "pearl_meta" / "model_card.json",
        ROOT / "data" / "models" / "ood_detector" / "model_card.json",
    ]
    required_keys = {"model_id", "version", "logic_summary", "thresholds", "performance_summary"}

    for card in cards:
        assert card.exists(), f"Missing model card: {card}"
        payload = json.loads(card.read_text(encoding="utf-8"))
        assert required_keys.issubset(payload.keys())


def test_day7_transition_report_and_handoff_exist():
    transition_report = ROOT / "docs" / "reports" / "regime_transition_test_report.md"
    handoff_doc = ROOT / "docs" / "reports" / "phase2_handoff" / "week2_regime_to_week3_sentiment_handoff.md"
    assert transition_report.exists()
    assert handoff_doc.exists()

    transition_text = transition_report.read_text(encoding="utf-8")
    assert "2008 Financial Crisis" in transition_text
    assert "2013 Taper Tantrum" in transition_text
    assert "2020 COVID Crash" in transition_text

    handoff_text = handoff_doc.read_text(encoding="utf-8")
    assert "regime_state" in handoff_text
    assert "risk_level" in handoff_text
    assert "Consensus Agent" in handoff_text


def test_day7_regime_output_stability():
    loader = _StaticLoader(_stable_frame())
    agent = RegimeAgent(loader=loader)

    pred1 = agent.detect_regime("RELIANCE.NS")
    pred2 = agent.detect_regime("RELIANCE.NS")

    assert pred1.regime_state == pred2.regime_state
    assert pred1.risk_level == pred2.risk_level
    assert pred1.transition_probability == pred2.transition_probability
    assert pred1.model_id == pred2.model_id

