from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.consensus.offline_pipeline import (
    PipelineConfig,
    aggregate_hourly_sentiment,
    build_market_frame,
    merge_macro_asof,
    run_walk_forward,
    run_weighted_consensus,
    select_consensus_training_rows,
)


def _synthetic_ohlcv(rows: int = 100, symbol: str = "TEST.NS") -> pd.DataFrame:
    rng = np.random.default_rng(123)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [base + timedelta(hours=i) for i in range(rows)]
    returns = rng.normal(0.0003, 0.01, rows)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": close * (1 + rng.normal(0.0, 0.001, rows)),
            "high": close * (1 + rng.normal(0.002, 0.001, rows)),
            "low": close * (1 - rng.normal(0.002, 0.001, rows)),
            "close": close,
            "volume": rng.integers(1000, 5000, rows),
            "interval": "1h",
        }
    )


def test_macro_asof_join_never_uses_future_rows():
    market = pd.DataFrame(
        {
            "symbol": ["TEST.NS", "TEST.NS"],
            "timestamp": pd.to_datetime(
                ["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"], utc=True
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1200],
        }
    )
    macro = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T09:30:00Z",  # valid as-of source
                    "2026-01-01T11:30:00Z",  # future for second row
                ],
                utc=True,
            ),
            "indicator_name": ["CPI", "CPI"],
            "value": [4.0, 9.0],
        }
    )

    joined, checks = merge_macro_asof(market, macro)

    assert checks["asof_safe_violations"] == 0
    assert list(joined["CPI"]) == [4.0, 4.0]


def test_walk_forward_produces_strict_oos_rows():
    ohlcv = _synthetic_ohlcv(rows=96, symbol="TEST.NS")
    market_frame = build_market_frame(ohlcv, tx_cost_bps=1.0, neutral_vol_scale=0.05)
    sentiment_hourly = pd.DataFrame(
        {
            "symbol": market_frame["symbol"],
            "timestamp": market_frame["timestamp"],
            "sentiment_score": 0.0,
            "sentiment_confidence": 0.0,
            "sentiment_source_count": 0,
            "sentiment_freshness": "missing",
            "sentiment_stale_flag": True,
            "sentiment_missing": True,
            "sentiment_dispersion": 0.0,
            "sentiment_novelty": 0.0,
        }
    )
    config = PipelineConfig(
        symbols=["TEST.NS"],
        output_root=Path("data/reports/training_runs"),
        min_train_rows=20,
        test_size_ratio=0.2,
    )

    base_df, _ = run_walk_forward({"TEST.NS": market_frame}, sentiment_hourly, config)

    assert not base_df.empty
    assert set(base_df["split"].unique()) <= {"train_oos", "test"}
    assert base_df["timestamp"].is_monotonic_increasing
    assert base_df["actual_return"].notna().all()


def test_consensus_training_rows_filter_is_oos_only():
    base_df = pd.DataFrame(
        {
            "split": ["train_oos", "train_oos", "test", "test"],
            "value": [1, 2, 3, 4],
        }
    )
    train_rows = select_consensus_training_rows(base_df)
    assert set(train_rows["split"].unique()) == {"train_oos"}
    assert train_rows["value"].tolist() == [1, 2]


def test_text_hourly_aggregation_is_asof_safe():
    market = _synthetic_ohlcv(rows=4, symbol="TEST.NS")
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market_frames = {"TEST.NS": market}

    doc_df = pd.DataFrame(
        {
            "symbol": ["TEST.NS"],
            "timestamp": [market["timestamp"].iloc[-1] + timedelta(hours=1)],  # strictly in the future
            "headline": ["Future document"],
            "sentiment_score": [0.9],
            "sentiment_confidence": [0.8],
        }
    )
    hourly = aggregate_hourly_sentiment(
        doc_df=doc_df,
        market_frames=market_frames,
        lookback_hours=24,
        stale_hours=6,
    )

    assert (hourly["sentiment_source_count"] == 0).all()
    assert hourly["sentiment_missing"].all()


def test_weighted_consensus_sentiment_fallback_behavior():
    base_df = pd.DataFrame(
        [
            {
                "symbol": "TEST.NS",
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "split": "test",
                "technical_score": 0.4,
                "regime_score": 0.2,
                "sentiment_score": 0.7,
                "technical_confidence": 0.7,
                "regime_confidence": 0.7,
                "sentiment_confidence": 0.7,
                "sentiment_missing": True,
                "sentiment_stale_flag": True,
                "regime_ood_warning": False,
                "regime_ood_alien": False,
                "agent_sign_disagreement_count": 0,
                "agent_score_dispersion": 0.1,
                "volatility": 0.01,
                "technical_volatility": 0.01,
            },
            {
                "symbol": "TEST.NS",
                "timestamp": pd.Timestamp("2026-01-01T01:00:00Z"),
                "split": "test",
                "technical_score": 0.4,
                "regime_score": 0.2,
                "sentiment_score": 0.7,
                "technical_confidence": 0.7,
                "regime_confidence": 0.7,
                "sentiment_confidence": 0.7,
                "sentiment_missing": False,
                "sentiment_stale_flag": True,
                "regime_ood_warning": False,
                "regime_ood_alien": False,
                "agent_sign_disagreement_count": 0,
                "agent_score_dispersion": 0.1,
                "volatility": 0.01,
                "technical_volatility": 0.01,
            },
            {
                "symbol": "TEST.NS",
                "timestamp": pd.Timestamp("2026-01-01T02:00:00Z"),
                "split": "test",
                "technical_score": 0.4,
                "regime_score": 0.2,
                "sentiment_score": 0.7,
                "technical_confidence": 0.7,
                "regime_confidence": 0.7,
                "sentiment_confidence": 0.7,
                "sentiment_missing": False,
                "sentiment_stale_flag": False,
                "regime_ood_warning": False,
                "regime_ood_alien": False,
                "agent_sign_disagreement_count": 0,
                "agent_score_dispersion": 0.1,
                "volatility": 0.01,
                "technical_volatility": 0.01,
            },
        ]
    )

    pred = run_weighted_consensus(base_df, tx_cost_bps=8.0, neutral_vol_scale=0.5)
    w_sent = pred["w_sentiment"].tolist()
    assert w_sent[0] < w_sent[1] < w_sent[2]
