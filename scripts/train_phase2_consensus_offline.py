from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.consensus.offline_pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Phase-2 training pipeline up to Consensus Agent."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE.NS", "TATASTEEL.NS", "USDINR=X"],
        help="Symbols to include (hourly).",
    )
    parser.add_argument(
        "--output-root",
        default="data/reports/training_runs",
        help="Root directory where run artifacts are written.",
    )
    parser.add_argument("--min-train-rows", type=int, default=60)
    parser.add_argument("--test-size-ratio", type=float, default=0.2)
    parser.add_argument("--tx-cost-bps", type=float, default=8.0)
    parser.add_argument("--neutral-vol-scale", type=float, default=0.5)
    parser.add_argument("--sentiment-lookback-hours", type=int, default=24)
    parser.add_argument("--sentiment-stale-hours", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-challenger", action="store_true")
    parser.add_argument("--silver-ohlcv-root", default="data/silver/ohlcv")
    parser.add_argument("--silver-macro-root", default="data/silver/macro")
    parser.add_argument(
        "--textual-canonical-path",
        default="docs/reports/day4_sync_s2/artifacts/textual_canonical_2026-03-05.parquet",
    )
    parser.add_argument(
        "--textual-sidecar-path",
        default="docs/reports/day4_sync_s2/artifacts/textual_sidecar_2026-03-05.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        symbols=args.symbols,
        output_root=Path(args.output_root),
        silver_ohlcv_root=Path(args.silver_ohlcv_root),
        silver_macro_root=Path(args.silver_macro_root),
        textual_canonical_path=Path(args.textual_canonical_path),
        textual_sidecar_path=Path(args.textual_sidecar_path),
        min_train_rows=args.min_train_rows,
        test_size_ratio=args.test_size_ratio,
        tx_cost_bps=args.tx_cost_bps,
        neutral_vol_scale=args.neutral_vol_scale,
        sentiment_lookback_hours=args.sentiment_lookback_hours,
        sentiment_stale_hours=args.sentiment_stale_hours,
        seed=args.seed,
        skip_challenger=args.skip_challenger,
    )
    result = run_pipeline(config)
    print(f"Run directory: {result['run_dir']}")
    print(f"Report: {result['report_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()
