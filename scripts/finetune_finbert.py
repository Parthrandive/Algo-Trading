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
    DEFAULT_REMOTE_DATASETS,
    DEFAULT_THRESHOLDS,
    HF_BACKEND,
    SKLEARN_BACKEND,
    huggingface_backend_available,
    load_dataset_rows,
    load_examples_from_sources,
    load_remote_dataset_sources,
    persist_training_artifact,
    train_hf_sentiment_model,
    train_sklearn_sentiment_model,
)


def _load_local_sources(dataset_specs: list[str]) -> dict[str, list[dict]]:
    sources: dict[str, list[dict]] = {}
    for item in dataset_specs:
        dataset_name, raw_path = item.split("=", 1)
        sources[dataset_name] = load_dataset_rows(Path(raw_path))
    return sources


def _parse_remote_specs(dataset_specs: list[str]) -> tuple[list[str], dict[str, str]]:
    dataset_names: list[str] = []
    split_overrides: dict[str, str] = {}
    for item in dataset_specs:
        dataset_name, split = item, None
        if "=" in item:
            dataset_name, split = item.split("=", 1)
        dataset_names.append(dataset_name)
        if split:
            split_overrides[dataset_name] = split
    return dataset_names, split_overrides


def _resolve_backend(requested_backend: str, *, has_remote_datasets: bool) -> str:
    if requested_backend in {HF_BACKEND, SKLEARN_BACKEND}:
        return requested_backend
    if has_remote_datasets and huggingface_backend_available():
        return HF_BACKEND
    return SKLEARN_BACKEND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sentiment artifact with a local sklearn fallback or HF fine-tune.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Local dataset mapping: dataset_name=path/to/file.(json|jsonl|csv). Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--remote-dataset",
        action="append",
        default=[],
        help="Remote dataset name, optionally dataset_name=split. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--include-default-remote-dataset",
        action="store_true",
        help="Include the default Indian-market remote dataset bundle.",
    )
    parser.add_argument(
        "--include-bootstrap-corpus",
        action="store_true",
        help="Include the synthetic bootstrap corpus for smoke training or hybrid experiments.",
    )
    parser.add_argument("--backend", choices=["auto", HF_BACKEND, SKLEARN_BACKEND], default="auto")
    parser.add_argument("--output-dir", default="data/models/sentiment/finbert_indian_v1")
    parser.add_argument("--model-id", default="finbert_indian_v1")
    parser.add_argument("--symbol", default="NSE_SENTIMENT")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--base-model-id", default="ProsusAI/finbert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face to download the base model and datasets instead of requiring local cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_sources = _load_local_sources(args.dataset)
    remote_specs = list(args.remote_dataset)
    if args.include_default_remote_dataset:
        for dataset_name in DEFAULT_REMOTE_DATASETS:
            if dataset_name not in remote_specs:
                remote_specs.append(dataset_name)
    remote_dataset_names, split_overrides = _parse_remote_specs(remote_specs)

    remote_sources: dict[str, list[dict]] = {}
    remote_dataset_sizes: dict[str, int] = {}
    if remote_dataset_names:
        remote_sources, remote_dataset_sizes = load_remote_dataset_sources(
            remote_dataset_names,
            split_overrides=split_overrides,
            sample_limit=args.sample_limit,
        )

    dataset_sources = {**local_sources, **remote_sources}
    if not dataset_sources and not args.include_bootstrap_corpus:
        raise SystemExit(
            "No datasets supplied. Use --dataset ..., --remote-dataset ..., or --include-bootstrap-corpus."
        )

    examples = load_examples_from_sources(dataset_sources, include_bootstrap=args.include_bootstrap_corpus)
    dataset_sizes = {name: len(rows) for name, rows in local_sources.items()}
    dataset_sizes.update(remote_dataset_sizes)
    if args.include_bootstrap_corpus:
        dataset_sizes["synthetic_bootstrap"] = len(BOOTSTRAP_CORPUS["synthetic_bootstrap"])

    selected_backend = _resolve_backend(args.backend, has_remote_datasets=bool(remote_dataset_names))
    if selected_backend == HF_BACKEND and not huggingface_backend_available():
        raise SystemExit(
            "Hugging Face dependencies are not available in this Python environment. "
            "Install them or use --backend sklearn_logistic_regression."
        )

    output_dir = Path(args.output_dir)
    synthetic_data = bool(args.include_bootstrap_corpus and not dataset_sources)
    if selected_backend == HF_BACKEND:
        artifact = train_hf_sentiment_model(
            examples,
            output_dir=output_dir,
            model_id=args.model_id,
            symbol=args.symbol,
            version=args.version,
            dataset_sizes=dataset_sizes,
            thresholds=DEFAULT_THRESHOLDS,
            synthetic_data=synthetic_data,
            base_model_id=args.base_model_id,
            seed=args.seed,
            val_ratio=args.val_ratio,
            max_length=args.max_length,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            local_files_only=not args.allow_downloads,
        )
        training_meta = json.loads(artifact.training_meta_path.read_text(encoding="utf-8"))
        training_report = training_meta["validation_metrics"]
    else:
        pipeline, training_report = train_sklearn_sentiment_model(
            examples,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )
        artifact = persist_training_artifact(
            output_dir=output_dir,
            pipeline=pipeline,
            model_id=args.model_id,
            symbol=args.symbol,
            version=args.version,
            training_report=training_report,
            dataset_sizes=dataset_sizes,
            thresholds=DEFAULT_THRESHOLDS,
            synthetic_data=synthetic_data,
        )

    summary = {
        "output_dir": str(artifact.output_dir),
        "training_meta": str(artifact.training_meta_path),
        "model_card": str(artifact.model_card_path),
        "validation_accuracy": training_report["accuracy"],
        "backend": training_report["backend"],
        "dataset_sizes": dataset_sizes,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
