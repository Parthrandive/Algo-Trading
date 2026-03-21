from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.agents.sentiment.datasets import (
    FinBERTFineTunePipeline,
    IndianSentimentDatasetLoader,
    SentimentTrainingExample,
)
from src.agents.sentiment.schemas import SentimentLabel

BOOTSTRAP_CORPUS: dict[str, list[dict[str, str]]] = {
    "synthetic_bootstrap": [
        {"text": "RBI rate cut boosts banking stocks and Nifty closes higher", "label": "positive"},
        {"text": "Strong earnings beat lifts Infosys and improves market mood", "label": "positive"},
        {"text": "SEBI clears compliance issue and shares rally on relief", "label": "positive"},
        {"text": "FII inflows strengthen rupee sentiment across Indian equities", "label": "positive"},
        {"text": "Government capex push supports infrastructure names and market breadth", "label": "positive"},
        {"text": "Crude spike hurts rupee outlook and drags import-heavy sectors", "label": "negative"},
        {"text": "SEBI investigation weighs on promoter credibility and stock demand", "label": "negative"},
        {"text": "Guidance cut triggers selloff in IT stocks after weak quarter", "label": "negative"},
        {"text": "RBI tightening surprise pressures rate-sensitive lenders and NBFCs", "label": "negative"},
        {"text": "Adverse audit finding raises fraud concerns around the small-cap rally", "label": "negative"},
        {"text": "Markets close mixed as traders wait for RBI commentary", "label": "neutral"},
        {"text": "Sensex trades range-bound with no clear directional catalyst", "label": "neutral"},
        {"text": "Management reiterates prior outlook and analysts stay cautious", "label": "neutral"},
        {"text": "Headline flow remains balanced with sector rotation across the index", "label": "neutral"},
        {"text": "Rupee holds steady while investors watch global yields", "label": "neutral"},
    ]
}

LABEL_ORDER = [SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL, SentimentLabel.POSITIVE]
LABEL_TO_ID = {
    SentimentLabel.NEGATIVE.value: 0,
    SentimentLabel.NEUTRAL.value: 1,
    SentimentLabel.POSITIVE.value: 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
SKLEARN_BACKEND = "sklearn_logistic_regression"
HF_BACKEND = "huggingface_transformers"
DEFAULT_REMOTE_DATASETS = ("harixn/indian_news_sentiment",)
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "positive": {"precision_min": 0.75, "recall_min": 0.72},
    "neutral": {"precision_min": 0.70, "recall_min": 0.68},
    "negative": {"precision_min": 0.78, "recall_min": 0.74},
}


class SentimentPredictor(Protocol):
    def predict(self, text: str) -> Any:
        ...


@dataclass(frozen=True)
class TrainingArtifact:
    output_dir: Path
    classifier_path: Path | None
    manifest_path: Path
    model_card_path: Path
    training_meta_path: Path


def _label_name(label: SentimentLabel) -> str:
    return label.value


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [dict(row) for row in payload["rows"] if isinstance(row, Mapping)]
        raise ValueError(f"Unsupported JSON dataset structure in {path}")

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            if isinstance(parsed, Mapping):
                rows.append(dict(parsed))
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    raise ValueError(f"Unsupported dataset file: {path}")


def huggingface_backend_available() -> bool:
    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _require_hf_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from datasets import Dataset, load_dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face training requires the 'transformers', 'datasets', and 'accelerate' packages."
        ) from exc
    return Dataset, load_dataset, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


def load_remote_dataset_rows(
    dataset_name: str,
    *,
    dataset_loader: IndianSentimentDatasetLoader | None = None,
    split: str | None = None,
    sample_limit: int | None = None,
) -> list[dict[str, Any]]:
    loader = dataset_loader or IndianSentimentDatasetLoader()
    spec = loader.dataset_specs.get(dataset_name)
    if spec is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    _, load_dataset, _, _, _, _, _ = _require_hf_modules()
    dataset_split = split or spec.split
    dataset = load_dataset(spec.dataset_id, split=dataset_split)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if sample_limit is not None and idx >= sample_limit:
            break
        if isinstance(row, Mapping):
            rows.append(dict(row))
            continue
        rows.append(dict(row.items()))
    return rows


def load_remote_dataset_sources(
    dataset_names: Iterable[str],
    *,
    dataset_loader: IndianSentimentDatasetLoader | None = None,
    split_overrides: Mapping[str, str] | None = None,
    sample_limit: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    loader = dataset_loader or IndianSentimentDatasetLoader()
    sources: dict[str, list[dict[str, Any]]] = {}
    dataset_sizes: dict[str, int] = {}
    for dataset_name in dataset_names:
        rows = load_remote_dataset_rows(
            dataset_name,
            dataset_loader=loader,
            split=(split_overrides or {}).get(dataset_name),
            sample_limit=sample_limit,
        )
        sources[dataset_name] = rows
        dataset_sizes[dataset_name] = len(rows)
    return sources, dataset_sizes


def load_examples_from_sources(
    dataset_sources: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    *,
    dataset_loader: IndianSentimentDatasetLoader | None = None,
    include_bootstrap: bool = False,
) -> list[SentimentTrainingExample]:
    loader = dataset_loader or IndianSentimentDatasetLoader()
    merged: list[SentimentTrainingExample] = []
    active_sources: dict[str, Iterable[Mapping[str, Any]]] = {}
    if dataset_sources:
        active_sources.update(dataset_sources)
    if include_bootstrap:
        active_sources.update(BOOTSTRAP_CORPUS)

    for dataset_name, rows in active_sources.items():
        if dataset_name in loader.dataset_specs:
            merged.extend(loader.load_examples(dataset_name, rows))
            continue

        ad_hoc_rows = []
        for row in rows:
            text = str(row.get("text") or "").strip()
            label = str(row.get("label") or row.get("sentiment") or "").strip()
            if text and label:
                ad_hoc_rows.append({"text": text, "label": label})
        merged.extend(loader.load_examples("SEntFiN", ad_hoc_rows))
    return merged


def build_training_batch(
    dataset_sources: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    *,
    include_bootstrap: bool = False,
    max_length: int = 96,
) -> dict[str, Any]:
    pipeline = FinBERTFineTunePipeline(max_length=max_length)
    merged_sources = dict(dataset_sources or {})
    if include_bootstrap:
        merged_sources.update(BOOTSTRAP_CORPUS)
    return pipeline.build_training_batch(merged_sources)


def _examples_to_xy(examples: Iterable[SentimentTrainingExample]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for example in examples:
        texts.append(example.text)
        labels.append(_label_name(example.label))
    return texts, labels


def _build_classification_report(
    *,
    texts: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    probabilities: np.ndarray | None = None,
    confidences: list[float] | None = None,
    backend: str,
    sample_size: int | None = None,
    train_size: int | None = None,
    validation_size: int | None = None,
    prediction_key: str = "predictions",
) -> dict[str, Any]:
    label_names = [_label_name(label) for label in LABEL_ORDER]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual_labels,
        predicted_labels,
        labels=label_names,
        zero_division=0,
    )
    report: dict[str, Any] = {
        "backend": backend,
        "accuracy": float(accuracy_score(actual_labels, predicted_labels)),
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "class_order": label_names,
        "confusion_matrix": confusion_matrix(actual_labels, predicted_labels, labels=label_names).tolist(),
        "classes": {},
        prediction_key: [],
    }
    if sample_size is not None:
        report["sample_size"] = int(sample_size)
    if train_size is not None:
        report["train_size"] = int(train_size)
    if validation_size is not None:
        report["validation_size"] = int(validation_size)

    for idx, label_name in enumerate(label_names):
        report["classes"][label_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    for idx, (text, predicted, actual) in enumerate(zip(texts, predicted_labels, actual_labels)):
        if confidences is not None:
            confidence = float(confidences[idx])
        elif probabilities is not None and idx < len(probabilities):
            confidence = float(np.max(probabilities[idx]))
        else:
            confidence = 0.5
        report[prediction_key].append(
            {
                "text": text,
                "predicted_label": str(predicted),
                "true_label": str(actual),
                "confidence": confidence,
            }
        )
    return report


def train_sklearn_sentiment_model(
    examples: list[SentimentTrainingExample],
    *,
    seed: int = 42,
    val_ratio: float = 0.25,
) -> tuple[Pipeline, dict[str, Any]]:
    if len(examples) < 6:
        raise ValueError("Need at least 6 labeled examples to train the sentiment model.")

    texts, labels = _examples_to_xy(examples)
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    stratify = labels if len(label_counts) >= 2 and min(label_counts.values()) >= 2 else None
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=400, random_state=seed)),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    probabilities = pipeline.predict_proba(val_texts)
    predicted_labels = [str(label) for label in pipeline.predict(val_texts)]
    report = _build_classification_report(
        texts=list(val_texts),
        actual_labels=list(val_labels),
        predicted_labels=predicted_labels,
        probabilities=np.asarray(probabilities),
        backend=SKLEARN_BACKEND,
        train_size=len(train_texts),
        validation_size=len(val_texts),
        prediction_key="validation_predictions",
    )
    return pipeline, report


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _build_training_arguments(
    training_arguments_cls: Any,
    *,
    output_dir: Path,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Any:
    common_kwargs = {
        "output_dir": str(output_dir / "hf_training_runs"),
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "report_to": [],
        "disable_tqdm": True,
        "remove_unused_columns": False,
    }
    for evaluation_key in ("evaluation_strategy", "eval_strategy"):
        try:
            return training_arguments_cls(**common_kwargs, **{evaluation_key: "epoch"})
        except TypeError:
            continue
    raise TypeError("Unable to construct Hugging Face TrainingArguments with the installed transformers version.")


def train_hf_sentiment_model(
    examples: list[SentimentTrainingExample],
    *,
    output_dir: Path,
    model_id: str,
    version: str,
    dataset_sizes: Mapping[str, int],
    thresholds: Mapping[str, Mapping[str, float]],
    synthetic_data: bool,
    base_model_id: str = "ProsusAI/finbert",
    seed: int = 42,
    val_ratio: float = 0.25,
    max_length: int = 96,
    num_train_epochs: float = 2.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    local_files_only: bool = True,
) -> TrainingArtifact:
    if len(examples) < 6:
        raise ValueError("Need at least 6 labeled examples to train the sentiment model.")

    Dataset, _, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed = (
        _require_hf_modules()
    )
    set_seed(seed)

    texts, labels = _examples_to_xy(examples)
    label_ids = [LABEL_TO_ID[label] for label in labels]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    stratify = labels if len(label_counts) >= 2 and min(label_counts.values()) >= 2 else None
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        label_ids,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=local_files_only, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: Mapping[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            list(batch["text"]),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    eval_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    eval_dataset = eval_dataset.map(tokenize_batch, batched=True)
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=len(LABEL_ORDER),
        id2label={idx: label for idx, label in ID_TO_LABEL.items()},
        label2id=LABEL_TO_ID,
        local_files_only=local_files_only,
    )
    if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        probabilities = _softmax(np.asarray(logits))
        predicted = probabilities.argmax(axis=1)
        actual = np.asarray(eval_pred.label_ids)
        report = _build_classification_report(
            texts=[""] * len(actual),
            actual_labels=[ID_TO_LABEL[int(label)] for label in actual],
            predicted_labels=[ID_TO_LABEL[int(label)] for label in predicted],
            probabilities=probabilities,
            backend=HF_BACKEND,
            sample_size=len(actual),
        )
        metrics = {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro_f1"]),
        }
        for label_name, class_metrics in report["classes"].items():
            metrics[f"{label_name}_precision"] = float(class_metrics["precision"])
            metrics[f"{label_name}_recall"] = float(class_metrics["recall"])
            metrics[f"{label_name}_f1"] = float(class_metrics["f1"])
        return metrics

    training_args = _build_training_arguments(
        TrainingArguments,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    prediction_output = trainer.predict(eval_dataset)
    logits = prediction_output.predictions[0] if isinstance(prediction_output.predictions, tuple) else prediction_output.predictions
    probabilities = _softmax(np.asarray(logits))
    predicted_ids = probabilities.argmax(axis=1)
    predicted_labels = [ID_TO_LABEL[int(label_id)] for label_id in predicted_ids]
    actual_labels = [ID_TO_LABEL[int(label_id)] for label_id in prediction_output.label_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    training_report = _build_classification_report(
        texts=list(val_texts),
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        backend=HF_BACKEND,
        train_size=len(train_texts),
        validation_size=len(val_texts),
        prediction_key="validation_predictions",
    )
    return persist_hf_training_artifact(
        output_dir=output_dir,
        model_id=model_id,
        version=version,
        training_report=training_report,
        dataset_sizes=dataset_sizes,
        thresholds=thresholds,
        synthetic_data=synthetic_data,
        base_model_id=base_model_id,
        hyperparameters={
            "max_length": max_length,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "local_files_only": local_files_only,
        },
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    examples: list[SentimentTrainingExample],
) -> dict[str, Any]:
    texts, labels = _examples_to_xy(examples)
    probabilities = pipeline.predict_proba(texts)
    predicted_labels = [str(label) for label in pipeline.predict(texts)]
    return _build_classification_report(
        texts=texts,
        actual_labels=labels,
        predicted_labels=predicted_labels,
        probabilities=np.asarray(probabilities),
        backend=SKLEARN_BACKEND,
        sample_size=len(texts),
    )


def evaluate_model(
    model: SentimentPredictor,
    examples: list[SentimentTrainingExample],
    *,
    backend: str,
) -> dict[str, Any]:
    texts, labels = _examples_to_xy(examples)
    predicted_labels: list[str] = []
    confidences: list[float] = []
    for text in texts:
        prediction = model.predict(text)
        label = getattr(prediction, "label", None)
        label_name = label.value if isinstance(label, SentimentLabel) else str(label or "neutral").lower()
        predicted_labels.append(label_name)
        confidences.append(float(getattr(prediction, "confidence", 0.5)))
    return _build_classification_report(
        texts=texts,
        actual_labels=labels,
        predicted_labels=predicted_labels,
        confidences=confidences,
        backend=backend,
        sample_size=len(texts),
    )


def threshold_report(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    status: dict[str, Any] = {"per_class": {}, "all_pass": True}
    metric_classes = metrics.get("classes", {})
    for label_name, threshold in thresholds.items():
        observed = metric_classes.get(label_name, {})
        precision_ok = float(observed.get("precision", 0.0)) >= float(threshold.get("precision_min", 0.0))
        recall_ok = float(observed.get("recall", 0.0)) >= float(threshold.get("recall_min", 0.0))
        status["per_class"][label_name] = {
            "precision_ok": precision_ok,
            "recall_ok": recall_ok,
        }
        status["all_pass"] = bool(status["all_pass"] and precision_ok and recall_ok)
    return status


def _write_artifact_metadata(
    *,
    output_dir: Path,
    classifier_path: Path | None,
    model_id: str,
    version: str,
    backend: str,
    training_report: Mapping[str, Any],
    dataset_sizes: Mapping[str, int],
    thresholds: Mapping[str, Mapping[str, float]],
    synthetic_data: bool,
    algorithm: str,
    description: str,
    hyperparameters: Mapping[str, Any],
    base_model_id: str | None = None,
) -> TrainingArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "artifact_manifest.json"
    model_card_path = output_dir / "model_card.json"
    training_meta_path = output_dir / "training_meta.json"
    threshold_status = threshold_report(training_report, thresholds)

    manifest = {
        "model_id": model_id,
        "version": version,
        "backend": backend,
        "class_order": [_label_name(label) for label in LABEL_ORDER],
        "synthetic_data": synthetic_data,
        "dataset_sizes": dict(dataset_sizes),
    }
    if base_model_id is not None:
        manifest["base_model_id"] = base_model_id
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    training_meta = {
        "model_id": model_id,
        "version": version,
        "backend": backend,
        "dataset_sizes": dict(dataset_sizes),
        "validation_metrics": dict(training_report),
        "threshold_status": threshold_status,
        "synthetic_data": synthetic_data,
        "hyperparameters": dict(hyperparameters),
    }
    if base_model_id is not None:
        training_meta["base_model_id"] = base_model_id
    training_meta_path.write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    model_card = {
        "model_id": model_id,
        "agent": "sentiment",
        "model_family": "finbert_adapted_runtime",
        "version": version,
        "owner": "Sentiment Agent",
        "algorithm": algorithm,
        "description": description,
        "hyperparameters": dict(hyperparameters),
        "performance": dict(training_report),
        "threshold_status": threshold_status,
        "synthetic_data": synthetic_data,
        "status": "bootstrap_only" if synthetic_data else "research_ready",
    }
    if base_model_id is not None:
        model_card["base_model_id"] = base_model_id
    model_card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    return TrainingArtifact(
        output_dir=output_dir,
        classifier_path=classifier_path,
        manifest_path=manifest_path,
        model_card_path=model_card_path,
        training_meta_path=training_meta_path,
    )


def persist_training_artifact(
    *,
    output_dir: Path,
    pipeline: Pipeline,
    model_id: str,
    version: str,
    training_report: Mapping[str, Any],
    dataset_sizes: Mapping[str, int],
    thresholds: Mapping[str, Mapping[str, float]],
    synthetic_data: bool,
) -> TrainingArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = output_dir / "classifier.pkl"
    with classifier_path.open("wb") as handle:
        pickle.dump(pipeline, handle)

    return _write_artifact_metadata(
        output_dir=output_dir,
        classifier_path=classifier_path,
        model_id=model_id,
        version=version,
        backend=SKLEARN_BACKEND,
        training_report=training_report,
        dataset_sizes=dataset_sizes,
        thresholds=thresholds,
        synthetic_data=synthetic_data,
        algorithm="FinBERT-compatible sentiment runtime with offline sklearn adaptation fallback",
        description=(
            "Uses Indian-market labeled text to adapt the sentiment runtime. "
            "When transformers assets are unavailable locally, the fallback backend remains clearly flagged."
        ),
        hyperparameters={
            "vectorizer": "tfidf_unigram_bigram",
            "classifier": "logistic_regression",
        },
    )


def persist_hf_training_artifact(
    *,
    output_dir: Path,
    model_id: str,
    version: str,
    training_report: Mapping[str, Any],
    dataset_sizes: Mapping[str, int],
    thresholds: Mapping[str, Mapping[str, float]],
    synthetic_data: bool,
    base_model_id: str,
    hyperparameters: Mapping[str, Any],
) -> TrainingArtifact:
    return _write_artifact_metadata(
        output_dir=output_dir,
        classifier_path=None,
        model_id=model_id,
        version=version,
        backend=HF_BACKEND,
        training_report=training_report,
        dataset_sizes=dataset_sizes,
        thresholds=thresholds,
        synthetic_data=synthetic_data,
        algorithm="FinBERT sequence classification fine-tune for Indian-market sentiment",
        description=(
            "Fine-tunes a FinBERT-compatible transformer on Indian-market labeled text. "
            "Artifact is intended for the slow lane and stays subject to promotion thresholds."
        ),
        hyperparameters=hyperparameters,
        base_model_id=base_model_id,
    )


def load_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_artifact_backend(output_dir: Path) -> str:
    manifest = load_artifact_manifest(output_dir)
    backend = manifest.get("backend")
    if isinstance(backend, str) and backend.strip():
        return backend
    if (output_dir / "classifier.pkl").exists():
        return SKLEARN_BACKEND
    if (output_dir / "config.json").exists():
        return HF_BACKEND
    raise FileNotFoundError(f"Could not determine artifact backend in {output_dir}")


def load_pipeline_artifact(output_dir: Path) -> Pipeline:
    classifier_path = output_dir / "classifier.pkl"
    with classifier_path.open("rb") as handle:
        pipeline = pickle.load(handle)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Unexpected classifier artifact at {classifier_path}")
    return pipeline
