from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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


@dataclass(frozen=True)
class TrainingArtifact:
    output_dir: Path
    classifier_path: Path
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
    predicted_labels = pipeline.predict(val_texts)

    label_names = [_label_name(label) for label in LABEL_ORDER]
    precision, recall, f1, support = precision_recall_fscore_support(
        val_labels,
        predicted_labels,
        labels=label_names,
        zero_division=0,
    )
    report = {
        "backend": "sklearn_logistic_regression",
        "train_size": len(train_texts),
        "validation_size": len(val_texts),
        "accuracy": float(accuracy_score(val_labels, predicted_labels)),
        "classes": {},
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "confusion_matrix": confusion_matrix(val_labels, predicted_labels, labels=label_names).tolist(),
        "class_order": label_names,
        "validation_predictions": [
            {
                "text": text,
                "predicted_label": str(predicted),
                "true_label": str(actual),
                "confidence": float(np.max(row_probs)),
            }
            for text, predicted, actual, row_probs in zip(val_texts, predicted_labels, val_labels, probabilities)
        ],
    }
    for idx, label_name in enumerate(label_names):
        report["classes"][label_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return pipeline, report


def evaluate_pipeline(
    pipeline: Pipeline,
    examples: list[SentimentTrainingExample],
) -> dict[str, Any]:
    texts, labels = _examples_to_xy(examples)
    probabilities = pipeline.predict_proba(texts)
    predicted_labels = pipeline.predict(texts)
    label_names = [_label_name(label) for label in LABEL_ORDER]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted_labels,
        labels=label_names,
        zero_division=0,
    )
    report = {
        "sample_size": len(texts),
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "class_order": label_names,
        "confusion_matrix": confusion_matrix(labels, predicted_labels, labels=label_names).tolist(),
        "classes": {},
        "predictions": [
            {
                "text": text,
                "predicted_label": str(predicted),
                "true_label": str(actual),
                "confidence": float(np.max(row_probs)),
            }
            for text, predicted, actual, row_probs in zip(texts, predicted_labels, labels, probabilities)
        ],
    }
    for idx, label_name in enumerate(label_names):
        report["classes"][label_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return report


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
    manifest_path = output_dir / "artifact_manifest.json"
    model_card_path = output_dir / "model_card.json"
    training_meta_path = output_dir / "training_meta.json"

    with classifier_path.open("wb") as handle:
        pickle.dump(pipeline, handle)

    manifest = {
        "model_id": model_id,
        "version": version,
        "backend": "sklearn_logistic_regression",
        "class_order": [_label_name(label) for label in LABEL_ORDER],
        "synthetic_data": synthetic_data,
        "dataset_sizes": dict(dataset_sizes),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    threshold_status = threshold_report(training_report, thresholds)
    training_meta = {
        "model_id": model_id,
        "version": version,
        "backend": "sklearn_logistic_regression",
        "dataset_sizes": dict(dataset_sizes),
        "validation_metrics": dict(training_report),
        "threshold_status": threshold_status,
        "synthetic_data": synthetic_data,
    }
    training_meta_path.write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    model_card = {
        "model_id": model_id,
        "agent": "sentiment",
        "model_family": "finbert_adapted_runtime",
        "version": version,
        "owner": "Sentiment Agent",
        "algorithm": "FinBERT-compatible sentiment runtime with offline sklearn adaptation fallback",
        "description": (
            "Uses Indian-market labeled text to adapt the sentiment runtime. "
            "When transformers assets are unavailable locally, the fallback backend remains clearly flagged."
        ),
        "hyperparameters": {
            "vectorizer": "tfidf_unigram_bigram",
            "classifier": "logistic_regression",
        },
        "performance": dict(training_report),
        "threshold_status": threshold_status,
        "synthetic_data": synthetic_data,
        "status": "bootstrap_only" if synthetic_data else "research_ready",
    }
    model_card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    return TrainingArtifact(
        output_dir=output_dir,
        classifier_path=classifier_path,
        manifest_path=manifest_path,
        model_card_path=model_card_path,
        training_meta_path=training_meta_path,
    )


def load_pipeline_artifact(output_dir: Path) -> Pipeline:
    classifier_path = output_dir / "classifier.pkl"
    with classifier_path.open("rb") as handle:
        pipeline = pickle.load(handle)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Unexpected classifier artifact at {classifier_path}")
    return pipeline
