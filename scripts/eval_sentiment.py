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
    evaluate_pipeline,
    load_dataset_rows,
    load_examples_from_sources,
    load_pipeline_artifact,
    threshold_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the sentiment artifact on labeled data.")
    parser.add_argument("--artifact-dir", default="data/models/sentiment/finbert_indian_v1")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset mapping in the form dataset_name=path/to/file.(json|jsonl|csv). Repeat for multiple datasets.",
    )
    parser.add_argument("--include-bootstrap-corpus", action="store_true")
    parser.add_argument("--output", default="data/reports/sentiment_eval/classification_report.json")
    return parser.parse_args()


def _load_sources(dataset_specs: list[str]) -> dict[str, list[dict]]:
    sources: dict[str, list[dict]] = {}
    for item in dataset_specs:
        dataset_name, raw_path = item.split("=", 1)
        sources[dataset_name] = load_dataset_rows(Path(raw_path))
    return sources


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline_artifact(Path(args.artifact_dir))
    sources = _load_sources(args.dataset)
    if not sources and not args.include_bootstrap_corpus:
        raise SystemExit("No datasets supplied. Use --dataset ... or --include-bootstrap-corpus.")

    examples = load_examples_from_sources(sources, include_bootstrap=args.include_bootstrap_corpus)
    report = evaluate_pipeline(pipeline, examples)
    report["threshold_status"] = threshold_report(
        report,
        {
            "positive": {"precision_min": 0.75, "recall_min": 0.72},
            "neutral": {"precision_min": 0.70, "recall_min": 0.68},
            "negative": {"precision_min": 0.78, "recall_min": 0.74},
        },
    )
    report["synthetic_data_only"] = bool(args.include_bootstrap_corpus and not sources)
    if args.include_bootstrap_corpus:
        report["bootstrap_rows"] = len(BOOTSTRAP_CORPUS["synthetic_bootstrap"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "accuracy": report["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
