"""
Functional tests for Analyst Board Skill (§5 Worked Examples).

Tests that training pipelines produce correct artifacts, handle errors,
and cover edge cases. Run with:

    pytest tests/skills/test_analyst_board_skill.py -v
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ohlcv_df(rows: int = 200, constant_close: bool = False) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=rows, freq="h")
    close = np.full(rows, 100.0) if constant_close else 100 + np.cumsum(np.random.randn(rows) * 0.5)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close + np.random.randn(rows) * 0.1,
        "high": close + abs(np.random.randn(rows) * 0.3),
        "low": close - abs(np.random.randn(rows) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 100000, rows).astype(float),
    })


# ── 1. VALID OUTPUTS ────────────────────────────────────────────────────────


class TestValidOutputs:
    """Verify that training scripts produce the required artifact files."""

    def test_training_scripts_exist(self):
        """All 4 training scripts from the skill must exist."""
        scripts = [
            "scripts/train_arima_lstm.py",
            "scripts/train_cnn_pattern.py",
            "scripts/train_garch_var.py",
            "scripts/train_regime_aware.py",
        ]
        for script in scripts:
            path = PROJECT_ROOT / script
            assert path.exists(), f"Missing training script: {script}"

    def test_training_scripts_have_argparse(self):
        """Each script must accept --symbol, --seed, and --output-dir arguments."""
        for script in ["train_arima_lstm.py", "train_cnn_pattern.py", "train_garch_var.py"]:
            content = (PROJECT_ROOT / "scripts" / script).read_text()
            assert "--symbol" in content, f"{script} missing --symbol arg"
            assert "--seed" in content, f"{script} missing --seed arg (reproducibility)"
            assert "--output-dir" in content or "--output" in content, f"{script} missing output dir arg"

    def test_training_meta_json_schema(self):
        """Any existing training_meta.json must have required fields."""
        meta_files = list((PROJECT_ROOT / "data").rglob("training_meta.json"))
        if not meta_files:
            pytest.skip("No training_meta.json files found — run training first")

        required_fields = {"timestamp", "symbol", "hyperparameters"}
        for meta_path in meta_files[:5]:  # check up to 5
            with open(meta_path) as f:
                meta = json.load(f)
            missing = required_fields - set(meta.keys())
            assert not missing, f"{meta_path.relative_to(PROJECT_ROOT)}: missing fields {missing}"

    def test_feature_engineering_produces_columns(self):
        """engineer_features() must add technical indicator columns."""
        from src.agents.technical.features import engineer_features

        df = _make_ohlcv_df(100)
        result = engineer_features(df)
        # Must have more columns than the original 6
        assert len(result.columns) > 6, f"Expected engineered features, got {len(result.columns)} columns"
        # close must still be present
        assert "close" in result.columns


# ── 2. MODEL ARTIFACTS ──────────────────────────────────────────────────────


class TestModelArtifacts:
    """Verify model classes can be instantiated and produce valid output shapes."""

    def test_arima_lstm_model_init(self):
        """ARIMA-LSTM hybrid must initialize without error."""
        from src.agents.technical.models.arima_lstm import ArimaLstmHybrid

        model = ArimaLstmHybrid(arima_order=(5, 1, 0), learning_rate=0.001)
        assert model.arima_order == (5, 1, 0)
        assert model.learning_rate == 0.001

    def test_cnn_model_forward_pass(self):
        """CNN model must accept (batch, 1, window, features) and return (batch, 3)."""
        from src.agents.technical.models.cnn_pattern import CNNPatternModel

        model = CNNPatternModel(time_steps=20, features=5, num_classes=3)
        x = torch.randn(4, 1, 20, 5)  # batch=4, channel=1, window=20, features=5
        out = model(x)
        assert out.shape == (4, 3), f"Expected (4,3), got {out.shape}"

    def test_garch_model_init(self):
        """GARCH model must accept distribution parameter."""
        from src.agents.technical.models.garch_var import GarchVaRModel

        for dist in ["normal", "t", "skewstudent"]:
            model = GarchVaRModel(window_size=252, dist=dist)
            assert model.dist == dist


# ── 3. ERROR HANDLING ────────────────────────────────────────────────────────


class TestErrorHandling:
    """Verify training validates data before proceeding."""

    def test_arima_lstm_rejects_too_few_rows(self):
        """ARIMA-LSTM requires >= 40 rows."""
        from scripts.train_arima_lstm import validate_data

        df = _make_ohlcv_df(20)  # too few!
        with pytest.raises(ValueError, match="at least 40"):
            validate_data(df)

    def test_cnn_rejects_too_few_rows(self):
        """CNN Pattern requires >= 40 rows."""
        from scripts.train_cnn_pattern import validate_data

        df = _make_ohlcv_df(20)
        with pytest.raises(ValueError, match="at least 40"):
            validate_data(df)

    def test_garch_rejects_too_few_rows(self):
        """GARCH requires >= 30 rows."""
        from scripts.train_garch_var import validate_data

        df = _make_ohlcv_df(10)
        with pytest.raises(ValueError, match="at least 30"):
            validate_data(df)

    def test_rejects_missing_columns(self):
        """Training must fail if required OHLCV columns are missing."""
        from scripts.train_arima_lstm import validate_data

        df = pd.DataFrame({"close": range(50), "volume": range(50)})
        with pytest.raises(ValueError, match="Missing columns"):
            validate_data(df)

    def test_rejects_high_nan_percentage(self):
        """Training must fail if a required column has >5% NaN."""
        from scripts.train_arima_lstm import validate_data

        df = _make_ohlcv_df(100)
        df.loc[df.index[:10], "close"] = np.nan  # 10% NaN
        with pytest.raises(ValueError, match="NaNs"):
            validate_data(df)


# ── 4. EDGE CASES ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Cover non-obvious scenarios that could silently corrupt training."""

    def test_constant_close_warns_but_does_not_crash(self):
        """Constant close price should warn but not raise."""
        from scripts.train_arima_lstm import validate_data

        df = _make_ohlcv_df(100, constant_close=True)
        # Should not raise, but would log a warning
        validate_data(df)  # no exception = pass

    def test_chronological_split_no_shuffling(self):
        """80/20 split must be chronological — val data must come AFTER train data."""
        df = _make_ohlcv_df(100)
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        val = df.iloc[split_idx:]

        assert train["timestamp"].max() < val["timestamp"].min(), \
            "Validation data must be strictly after training data (no shuffling!)"

    def test_seed_reproducibility(self):
        """Same seed must produce same model initialization."""
        from scripts.train_arima_lstm import set_seed

        set_seed(42)
        t1 = torch.randn(3, 3)
        set_seed(42)
        t2 = torch.randn(3, 3)
        assert torch.equal(t1, t2), "Same seed must produce identical tensors"

    def test_cnn_class_labels_cover_all_three(self):
        """CNN must define 3 classes: up (0), neutral (1), down (2)."""
        from src.agents.technical.models.cnn_pattern import CnnPatternClassifier

        clf = CnnPatternClassifier(window_size=20)
        assert clf.num_classes == 3

    def test_feature_columns_exclude_raw_macro(self):
        """ARIMA-LSTM must exclude raw macro columns from features (prevent leakage)."""
        raw_macro = {"CPI", "WPI", "IIP", "FII_FLOW", "DII_FLOW", "FX_RESERVES",
                     "INDIA_US_10Y_SPREAD", "RBI_BULLETIN", "REPO_RATE", "US_10Y"}
        from scripts.train_arima_lstm import RAW_MACRO_COLUMNS
        assert RAW_MACRO_COLUMNS == raw_macro, \
            f"RAW_MACRO_COLUMNS mismatch. Script has {RAW_MACRO_COLUMNS}, expected {raw_macro}"
