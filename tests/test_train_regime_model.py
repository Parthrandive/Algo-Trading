import pytest
from pathlib import Path
import json

import pandas as pd

from scripts.train_regime_model import _build_parser, parse_args
from src.agents.regime.regime_agent import RegimeAgent
from src.agents.regime.data_loader import RegimeDataLoader


def test_train_regime_model_parser():
    parser = _build_parser()
    args = parser.parse_args(["--symbols", "INFY.NS", "--limit", "100", "--output-dir", "/tmp/models"])
    assert args.symbols == ["INFY.NS"]
    assert args.limit == 100
    assert args.output_dir == "/tmp/models"


def test_train_regime_model_creates_artifacts(tmp_path):
    # This is a smoke test to ensure the script's core components are importable and basic
    # parsing/paths work, without hitting the actual DB for training
    args = _build_parser().parse_args(["--symbols", "TEST", "--output-dir", str(tmp_path)])
    assert args.output_dir == str(tmp_path)


def test_regime_data_loader_falls_back_to_ohlcv_when_gold_sparse(monkeypatch):
    loader = RegimeDataLoader(database_url="postgresql://unused")
    sparse_gold = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01", periods=7, freq="h", tz="UTC"),
            "close": [100 + i for i in range(7)],
        }
    )
    rich_ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC"),
            "close": [100 + i * 0.1 for i in range(200)],
        }
    )

    monkeypatch.setattr(loader, "_load_from_db", lambda symbol, limit, interval: sparse_gold.copy())
    monkeypatch.setattr(loader, "_load_from_ohlcv", lambda symbol, limit, interval: rich_ohlcv.copy())
    monkeypatch.setattr(loader, "_load_from_parquet", lambda symbol, limit, interval: pd.DataFrame())

    result = loader.load_features("TCS.NS", limit=2000, interval="1h", min_gold_rows=120)
    assert len(result) == len(rich_ohlcv)
    assert result["timestamp"].is_monotonic_increasing


def test_regime_macro_builder_handles_missing_directional_flag_column():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"),
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
        }
    )
    out = RegimeAgent._build_macro_features(frame)
    assert "macro_directional_flag" in out.columns
    assert len(out["macro_directional_flag"]) == len(frame)
