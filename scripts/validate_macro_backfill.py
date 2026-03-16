import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.technical.data_loader import DataLoader, MACRO_COLUMNS


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate macro join coverage and train-ready gating on joined market bars."
    )
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Symbol to validate")
    parser.add_argument("--interval", default="1h", help="Bar interval (default: 1h)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max bar rows")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=float(os.getenv("MACRO_FEATURE_COVERAGE_THRESHOLD", "0.60")),
        help="Coverage ratio threshold (0-1) for train-ready gating",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output JSON path (default under data/reports)",
    )
    return parser.parse_args()


def _build_transition_samples(df: pd.DataFrame) -> dict[str, list[dict]]:
    samples: dict[str, list[dict]] = {}
    for feature in MACRO_COLUMNS:
        if feature not in df.columns:
            continue
        first_idx = df[feature].first_valid_index()
        if first_idx is None:
            continue
        start = max(0, int(first_idx) - 2)
        end = min(len(df), int(first_idx) + 3)
        snippet = df.iloc[start:end][["timestamp", feature]].copy()
        snippet["timestamp"] = pd.to_datetime(snippet["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        samples[feature] = snippet.to_dict(orient="records")
    return samples


def main() -> int:
    args = _parse_args()
    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")

    loader = DataLoader(db_url, macro_coverage_threshold=args.coverage_threshold)
    df = loader.load_historical_bars(
        symbol=args.symbol,
        limit=args.limit,
        interval=args.interval,
        include_macro=True,
        use_nse_fallback=False,
    )

    report = loader.last_macro_quality_report
    samples = _build_transition_samples(df)

    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(
        PROJECT_ROOT
        / "data"
        / "reports"
        / f"macro_backfill_validation_{args.symbol.replace('.', '_')}_{args.interval}_{now_str}.json"
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "interval": args.interval,
        "bars": int(len(df)),
        "coverage_threshold": args.coverage_threshold,
        "excluded_features": loader.last_macro_excluded_features,
        "macro_feature_report": report,
        "release_aware_join_samples": samples,
    }
    payload = _json_safe(payload)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"Validation report written: {output_path}")
    if report:
        print("\nFeature summary:")
        for feature in MACRO_COLUMNS:
            stats = report.get(feature, {})
            print(
                f"- {feature:20s} | rows={stats.get('row_count', 0):5d} | "
                f"coverage={stats.get('coverage_pct', 0.0):6.2f}% | "
                f"missing={stats.get('missing_after_join', 0):6d} | "
                f"train_ready={stats.get('train_ready', False)}"
            )
    else:
        print("No macro report produced (macro join may be unavailable).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
