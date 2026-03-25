from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_dataset_builder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sentiment.datasets import (
    DEFAULT_DATASET_SPECS,
    IndianDatasetSpec,
    IndianSentimentDatasetLoader,
    SentimentTrainingExample,
)
from src.agents.sentiment.schemas import SentimentLabel

LOGGER = logging.getLogger("finetune_finbert")


LABEL_TO_ID: dict[SentimentLabel, int] = {
    SentimentLabel.NEGATIVE: 0,
    SentimentLabel.NEUTRAL: 1,
    SentimentLabel.POSITIVE: 2,
}
ID_TO_LABEL_NAME: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
LABEL_NAME_TO_ID: dict[str, int] = {v: k for k, v in ID_TO_LABEL_NAME.items()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ProsusAI/finbert on Indian-market sentiment datasets.")
    parser.add_argument(
        "--runtime-config",
        default=str(PROJECT_ROOT / "configs" / "sentiment_agent_runtime_v1.json"),
        help="Path to sentiment runtime config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "models" / "sentiment" / "finbert_indian_v1"),
        help="Directory where the fine-tuned model and metadata will be saved.",
    )
    parser.add_argument(
        "--report-path",
        default=str(PROJECT_ROOT / "data" / "reports" / "sentiment_eval" / "classification_report.json"),
        help="JSON path for evaluation report.",
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of fine-tuning epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Per-device eval batch size.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional cap on each dataset before merging.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train split after splitting.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on validation split after splitting.",
    )
    parser.add_argument(
        "--allow-partial-datasets",
        action="store_true",
        help="Continue training if one or more configured datasets fail to load.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from local HuggingFace cache only.",
    )
    parser.add_argument(
        "--force-remote-model-download",
        action="store_true",
        help="Override runtime config and allow remote model/tokenizer download.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=str(PROJECT_ROOT / ".hf_cache"),
        help="Workspace-local HuggingFace cache directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load/validate datasets and print stats; do not train.",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU training (disables MPS/CUDA use in Trainer).",
    )
    return parser


def _setup_logger() -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


def _read_runtime_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_label_map_for_unknown_dataset() -> dict[str, SentimentLabel]:
    return {
        "-1": SentimentLabel.NEGATIVE,
        "0": SentimentLabel.NEUTRAL,
        "1": SentimentLabel.POSITIVE,
        "2": SentimentLabel.POSITIVE,
        "negative": SentimentLabel.NEGATIVE,
        "neutral": SentimentLabel.NEUTRAL,
        "positive": SentimentLabel.POSITIVE,
    }


def _merge_dataset_specs(runtime_config: dict[str, Any]) -> dict[str, IndianDatasetSpec]:
    runtime_datasets = runtime_config.get("datasets", {})
    specs = dict(DEFAULT_DATASET_SPECS)
    for dataset_name, ds_cfg in runtime_datasets.items():
        existing = specs.get(dataset_name)
        label_map = existing.label_map if existing is not None else _default_label_map_for_unknown_dataset()
        specs[dataset_name] = IndianDatasetSpec(
            dataset_id=str(ds_cfg.get("dataset_id", dataset_name)),
            split=str(ds_cfg.get("split", "train")),
            text_field=str(ds_cfg.get("text_field", "text")),
            label_field=str(ds_cfg.get("label_field", "label")),
            label_map=label_map,
        )
    return specs


def _label_to_id(label: SentimentLabel) -> int:
    return LABEL_TO_ID[label]


def _collect_dataset_license(dataset_id: str) -> str | None:
    try:
        builder = load_dataset_builder(dataset_id)
        return str(builder.info.license) if builder.info.license else None
    except Exception:
        return None


def _resolve_text_and_label_fields(
    rows: list[dict[str, Any]],
    *,
    text_field: str,
    label_field: str,
) -> tuple[str, str]:
    if not rows:
        return text_field, label_field

    sample = rows[0]
    keys = set(sample.keys())
    resolved_text = text_field
    resolved_label = label_field

    if resolved_text not in keys:
        for candidate in ("text", "headline", "content"):
            if candidate in keys:
                resolved_text = candidate
                break

    if resolved_label not in keys:
        for candidate in ("label", "sentiment", "target"):
            if candidate in keys:
                resolved_label = candidate
                break

    return resolved_text, resolved_label


def _load_examples(
    runtime_config: dict[str, Any],
    *,
    max_samples_per_dataset: int | None,
    allow_partial_datasets: bool,
    hf_cache_dir: str | None,
) -> tuple[list[SentimentTrainingExample], dict[str, Any]]:
    dataset_specs = _merge_dataset_specs(runtime_config)
    loader = IndianSentimentDatasetLoader(dataset_specs=dataset_specs)
    configured = runtime_config.get("datasets", {})
    if not configured:
        raise ValueError("No datasets configured in runtime config.")

    all_examples: list[SentimentTrainingExample] = []
    dataset_details: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for dataset_name, ds_cfg in configured.items():
        spec = dataset_specs[dataset_name]
        dataset_id = spec.dataset_id
        split = spec.split
        try:
            LOGGER.info("Loading dataset %s (split=%s) ...", dataset_id, split)
            hf_ds = load_dataset(dataset_id, split=split, cache_dir=hf_cache_dir)
            if max_samples_per_dataset is not None and max_samples_per_dataset > 0 and len(hf_ds) > max_samples_per_dataset:
                hf_ds = hf_ds.select(range(max_samples_per_dataset))
            rows = hf_ds.to_list()

            resolved_text_field, resolved_label_field = _resolve_text_and_label_fields(
                rows,
                text_field=spec.text_field,
                label_field=spec.label_field,
            )
            effective_spec = IndianDatasetSpec(
                dataset_id=spec.dataset_id,
                split=spec.split,
                text_field=resolved_text_field,
                label_field=resolved_label_field,
                label_map=spec.label_map,
            )
            effective_loader = IndianSentimentDatasetLoader(
                dataset_specs={dataset_name: effective_spec},
            )
            examples = effective_loader.load_examples(dataset_name, rows)
            all_examples.extend(examples)

            public_url = f"https://huggingface.co/datasets/{dataset_id}"
            dataset_details[dataset_name] = {
                "dataset_id": dataset_id,
                "split": split,
                "raw_rows": int(len(rows)),
                "usable_rows": int(len(examples)),
                "text_field": resolved_text_field,
                "label_field": resolved_label_field,
                "license": _collect_dataset_license(dataset_id),
                "public_availability_justification": f"Public dataset registry: {public_url}",
            }
            LOGGER.info(
                "Loaded %s: raw_rows=%s usable_rows=%s",
                dataset_id,
                len(rows),
                len(examples),
            )
        except Exception as exc:
            errors[dataset_name] = str(exc)
            LOGGER.error("Failed loading dataset %s: %s", dataset_name, exc)
            if not allow_partial_datasets:
                raise

    if not all_examples:
        raise RuntimeError(f"No usable training examples found. Dataset errors={errors}")

    summary = {
        "configured_datasets": list(configured.keys()),
        "dataset_details": dataset_details,
        "dataset_errors": errors,
        "total_examples": len(all_examples),
    }
    return all_examples, summary


def _split_examples(
    examples: list[SentimentTrainingExample],
    *,
    validation_ratio: float,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[list[SentimentTrainingExample], list[SentimentTrainingExample]]:
    if len(examples) < 30:
        raise ValueError(f"Need at least 30 examples to fine-tune; got {len(examples)}.")

    labels = [_label_to_id(x.label) for x in examples]
    stratify_labels = labels if len(set(labels)) >= 2 else None

    try:
        train_examples, val_examples = train_test_split(
            examples,
            test_size=validation_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError:
        train_examples, val_examples = train_test_split(
            examples,
            test_size=validation_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    if max_train_samples is not None and max_train_samples > 0 and len(train_examples) > max_train_samples:
        train_examples = train_examples[:max_train_samples]
    if max_val_samples is not None and max_val_samples > 0 and len(val_examples) > max_val_samples:
        val_examples = val_examples[:max_val_samples]

    return train_examples, val_examples


def _to_hf_dataset(examples: list[SentimentTrainingExample]) -> HFDataset:
    return HFDataset.from_dict(
        {
            "text": [x.text for x in examples],
            "labels": [_label_to_id(x.label) for x in examples],
        }
    )


def _build_metrics_fn():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            labels=[0, 1, 2],
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision_negative": float(precision[0]),
            "recall_negative": float(recall[0]),
            "f1_negative": float(f1[0]),
            "precision_neutral": float(precision[1]),
            "recall_neutral": float(recall[1]),
            "f1_neutral": float(f1[1]),
            "precision_positive": float(precision[2]),
            "recall_positive": float(recall[2]),
            "f1_positive": float(f1[2]),
            "f1_macro": float(np.mean(f1)),
        }

    return compute_metrics


def _build_training_args(**kwargs) -> TrainingArguments:
    if "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    dropped = sorted(set(kwargs.keys()) - set(filtered_kwargs.keys()))
    if dropped:
        LOGGER.info("Ignoring unsupported TrainingArguments keys: %s", dropped)
    return TrainingArguments(**filtered_kwargs)


def _threshold_checks(runtime_config: dict[str, Any], eval_metrics: dict[str, float]) -> dict[str, Any]:
    threshold_cfg = runtime_config.get("precision_recall_thresholds", {})
    checks: dict[str, Any] = {}
    all_pass = True
    for label_name, cfg in threshold_cfg.items():
        precision_key = f"eval_precision_{label_name}"
        recall_key = f"eval_recall_{label_name}"
        precision_value = float(eval_metrics.get(precision_key, 0.0))
        recall_value = float(eval_metrics.get(recall_key, 0.0))
        precision_min = float(cfg.get("precision_min", 0.0))
        recall_min = float(cfg.get("recall_min", 0.0))
        precision_pass = precision_value >= precision_min
        recall_pass = recall_value >= recall_min
        label_pass = precision_pass and recall_pass
        checks[label_name] = {
            "precision": precision_value,
            "recall": recall_value,
            "precision_min": precision_min,
            "recall_min": recall_min,
            "pass": label_pass,
        }
        all_pass = all_pass and label_pass
    checks["overall_pass"] = all_pass
    return checks


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)


def main() -> int:
    _setup_logger()
    args = _build_parser().parse_args()

    runtime_config_path = Path(args.runtime_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()

    runtime_config = _read_runtime_config(runtime_config_path)
    model_id = str(runtime_config.get("base_model", {}).get("model_id", "ProsusAI/finbert"))
    max_length = int(runtime_config.get("base_model", {}).get("max_length", 96))
    runtime_local_only = bool(runtime_config.get("base_model", {}).get("local_files_only", False))
    local_files_only = bool(args.local_files_only or runtime_local_only)
    if args.force_remote_model_download:
        local_files_only = False
    hf_cache_dir = str(Path(args.hf_cache_dir).resolve())
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_DATASETS_CACHE"] = str(Path(hf_cache_dir) / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_cache_dir) / "hub")
    Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))

    LOGGER.info("Runtime config: %s", runtime_config_path)
    LOGGER.info("Model: %s", model_id)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Local files only: %s", local_files_only)
    LOGGER.info("HF cache dir: %s", hf_cache_dir)

    examples, data_summary = _load_examples(
        runtime_config,
        max_samples_per_dataset=args.max_samples_per_dataset,
        allow_partial_datasets=bool(args.allow_partial_datasets),
        hf_cache_dir=hf_cache_dir,
    )

    train_examples, val_examples = _split_examples(
        examples,
        validation_ratio=float(args.validation_ratio),
        seed=int(args.seed),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    label_counts_total = {
        label_name: 0
        for label_name in ("negative", "neutral", "positive")
    }
    for ex in examples:
        label_counts_total[ex.label.value] += 1

    split_summary = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "total_examples": len(examples),
        "label_counts_total": label_counts_total,
    }
    LOGGER.info(
        "Split summary: train=%s val=%s total=%s",
        split_summary["train_examples"],
        split_summary["val_examples"],
        split_summary["total_examples"],
    )

    if args.dry_run:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "dry_run",
            "runtime_config": str(runtime_config_path),
            "model_id": model_id,
            "data_summary": data_summary,
            "split_summary": split_summary,
        }
        _write_json(report_path, payload)
        LOGGER.info("Dry run completed. Report: %s", report_path)
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=local_files_only,
        cache_dir=hf_cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        id2label=ID_TO_LABEL_NAME,
        label2id=LABEL_NAME_TO_ID,
        local_files_only=local_files_only,
        cache_dir=hf_cache_dir,
    )

    hf_train = _to_hf_dataset(train_examples)
    hf_val = _to_hf_dataset(val_examples)

    def tokenize_fn(batch: dict[str, list[str]]):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized_train = hf_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val = hf_val.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = _build_training_args(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=int(args.seed),
        dataloader_num_workers=0,
        do_train=True,
        do_eval=True,
        use_cpu=bool(args.use_cpu),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_build_metrics_fn(),
    )

    train_output = trainer.train()
    eval_metrics = trainer.evaluate()
    pred_output = trainer.predict(tokenized_val)

    val_labels = np.asarray(pred_output.label_ids, dtype=int)
    val_preds = np.argmax(pred_output.predictions, axis=1)
    cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2]).tolist()

    threshold_checks = _threshold_checks(runtime_config, eval_metrics)

    # Save model artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_timestamp = datetime.now(UTC).isoformat()
    training_meta = {
        "timestamp": run_timestamp,
        "symbol": "SENTIMENT_AGENT",
        "model": model_id,
        "hyperparameters": {
            "epochs": float(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "train_batch_size": int(args.train_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "validation_ratio": float(args.validation_ratio),
            "seed": int(args.seed),
            "max_length": max_length,
            "local_files_only": local_files_only,
            "hf_cache_dir": hf_cache_dir,
        },
        "dataset_summary": data_summary,
        "split_summary": split_summary,
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "threshold_checks": threshold_checks,
    }
    _write_json(output_dir / "training_meta.json", training_meta)

    report = {
        "timestamp": run_timestamp,
        "model_id": model_id,
        "runtime_config": str(runtime_config_path),
        "output_dir": str(output_dir),
        "train_runtime_seconds": float(train_output.metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(train_output.metrics.get("train_samples_per_second", 0.0)),
        "data_summary": data_summary,
        "split_summary": split_summary,
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "confusion_matrix": {
            "labels": ["negative", "neutral", "positive"],
            "matrix": cm,
        },
        "threshold_checks": threshold_checks,
        "status": "pass" if threshold_checks.get("overall_pass") else "fail",
    }
    _write_json(report_path, report)

    LOGGER.info("Saved fine-tuned model: %s", output_dir)
    LOGGER.info("Saved training metadata: %s", output_dir / "training_meta.json")
    LOGGER.info("Saved evaluation report: %s", report_path)
    LOGGER.info("Threshold overall pass: %s", threshold_checks.get("overall_pass"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
