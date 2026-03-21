from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sentiment.training import (
    BOOTSTRAP_CORPUS,
    load_dataset_rows,
    load_examples_from_sources,
    persist_training_artifact,
    train_sklearn_sentiment_model,
)


def _load_sources(dataset_specs: list[str]) -> dict[str, list[dict]]:
    sources: dict[str, list[dict]] = {}
    for item in dataset_specs:
        dataset_name, raw_path = item.split("=", 1)
        sources[dataset_name] = load_dataset_rows(Path(raw_path))
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline-safe FinBERT adaptation workflow for the sentiment agent.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset mapping in the form dataset_name=path/to/file.(json|jsonl|csv). Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--include-bootstrap-corpus",
        action="store_true",
        help="Include the synthetic bootstrap corpus for offline smoke training.",
    )
    parser.add_argument("--output-dir", default="data/models/sentiment/finbert_indian_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = _load_sources(args.dataset)
    if not sources and not args.include_bootstrap_corpus:
        raise SystemExit("No datasets supplied. Use --dataset ... or --include-bootstrap-corpus.")

    examples = load_examples_from_sources(sources, include_bootstrap=args.include_bootstrap_corpus)
    dataset_sizes = {name: len(rows) for name, rows in sources.items()}
    if args.include_bootstrap_corpus:
        dataset_sizes["synthetic_bootstrap"] = len(BOOTSTRAP_CORPUS["synthetic_bootstrap"])

    pipeline, training_report = train_sklearn_sentiment_model(
        examples,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    artifact = persist_training_artifact(
        output_dir=Path(args.output_dir),
        pipeline=pipeline,
        model_id="finbert_indian_v1",
        version="1.0",
        training_report=training_report,
        dataset_sizes=dataset_sizes,
        thresholds={
            "positive": {"precision_min": 0.75, "recall_min": 0.72},
            "neutral": {"precision_min": 0.70, "recall_min": 0.68},
            "negative": {"precision_min": 0.78, "recall_min": 0.74},
        },
        synthetic_data=args.include_bootstrap_corpus and not sources,
    )
    summary = {
        "output_dir": str(artifact.output_dir),
        "training_meta": str(artifact.training_meta_path),
        "model_card": str(artifact.model_card_path),
        "validation_accuracy": training_report["accuracy"],
        "backend": training_report["backend"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
