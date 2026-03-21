from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sentiment.sentiment_agent import SentimentAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nightly sentiment batch and persist scores.")
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--runtime-config", default="configs/sentiment_agent_runtime_v1.json")
    parser.add_argument("--output", default="data/reports/sentiment_eval/nightly_batch_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = SentimentAgent.from_default_components(
        Path(args.runtime_config),
        database_url=args.database_url,
        persist_predictions=True,
    )
    batch = agent.run_nightly_batch(lookback_hours=args.lookback_hours)

    summary = {
        "started_at_utc": batch.started_at_utc.isoformat(),
        "completed_at_utc": batch.completed_at_utc.isoformat(),
        "lookback_hours": batch.lookback_hours,
        "document_predictions": len(batch.document_predictions),
        "symbol_aggregates": [
            {
                "symbol": aggregate.symbol,
                "z_t": aggregate.z_t,
                "sample_size": aggregate.sample_size,
                "quality_status": aggregate.quality_status.value,
            }
            for aggregate in batch.symbol_aggregates
        ],
        "market_aggregate": (
            None
            if batch.market_aggregate is None
            else {
                "z_t": batch.market_aggregate.z_t,
                "sample_size": batch.market_aggregate.sample_size,
                "quality_status": batch.market_aggregate.quality_status.value,
            }
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "document_predictions": len(batch.document_predictions)}, indent=2))


if __name__ == "__main__":
    main()
