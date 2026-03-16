from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from src.agents.sentiment.schemas import SentimentLabel


@dataclass(frozen=True)
class SentimentTrainingExample:
    text: str
    label: SentimentLabel


@dataclass(frozen=True)
class IndianDatasetSpec:
    dataset_id: str
    split: str
    text_field: str
    label_field: str
    label_map: Mapping[str, SentimentLabel]


DEFAULT_DATASET_SPECS: dict[str, IndianDatasetSpec] = {
    "harixn/indian_news_sentiment": IndianDatasetSpec(
        dataset_id="harixn/indian_news_sentiment",
        split="train",
        text_field="text",
        label_field="sentiment",
        label_map={
            "0": SentimentLabel.NEGATIVE,
            "1": SentimentLabel.NEUTRAL,
            "2": SentimentLabel.POSITIVE,
            "negative": SentimentLabel.NEGATIVE,
            "neutral": SentimentLabel.NEUTRAL,
            "positive": SentimentLabel.POSITIVE,
        },
    ),
    "SEntFiN": IndianDatasetSpec(
        dataset_id="SEntFiN",
        split="train",
        text_field="text",
        label_field="label",
        label_map={
            "-1": SentimentLabel.NEGATIVE,
            "0": SentimentLabel.NEUTRAL,
            "1": SentimentLabel.POSITIVE,
            "negative": SentimentLabel.NEGATIVE,
            "neutral": SentimentLabel.NEUTRAL,
            "positive": SentimentLabel.POSITIVE,
        },
    ),
}


def _label_to_id(label: SentimentLabel) -> int:
    if label == SentimentLabel.NEGATIVE:
        return 0
    if label == SentimentLabel.NEUTRAL:
        return 1
    return 2


class BatchTokenizer(Protocol):
    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int = 64,
    ) -> Mapping[str, list[list[int]]]:
        ...


class SimpleWhitespaceTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    @staticmethod
    def _token_to_id(token: str) -> int:
        checksum = sum(token.encode("utf-8"))
        return 1000 + (checksum % 20000)

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int = 64,
    ) -> Mapping[str, list[list[int]]]:
        encoded_ids: list[list[int]] = []
        encoded_masks: list[list[int]] = []

        for text in texts:
            tokens = [token for token in text.lower().split() if token]
            token_ids = [self._token_to_id(token) for token in tokens]
            if truncation:
                token_ids = token_ids[: max(0, max_length - 2)]

            row = [self.cls_token_id] + token_ids + [self.sep_token_id]
            mask = [1] * len(row)

            if padding == "max_length" and len(row) < max_length:
                pad = max_length - len(row)
                row = row + [self.pad_token_id] * pad
                mask = mask + [0] * pad

            encoded_ids.append(row[:max_length])
            encoded_masks.append(mask[:max_length])

        return {"input_ids": encoded_ids, "attention_mask": encoded_masks}


class IndianSentimentDatasetLoader:
    def __init__(self, dataset_specs: Mapping[str, IndianDatasetSpec] | None = None):
        self.dataset_specs: Mapping[str, IndianDatasetSpec] = dataset_specs or DEFAULT_DATASET_SPECS

    def load_examples(
        self,
        dataset_name: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> list[SentimentTrainingExample]:
        spec = self._get_dataset_spec(dataset_name)
        examples: list[SentimentTrainingExample] = []

        for row in rows:
            raw_text = row.get(spec.text_field)
            raw_label = row.get(spec.label_field)

            if not isinstance(raw_text, str):
                continue
            text = raw_text.strip()
            if not text:
                continue

            label = self._normalize_label(raw_label, spec.label_map)
            if label is None:
                continue

            examples.append(SentimentTrainingExample(text=text, label=label))

        return examples

    def _get_dataset_spec(self, dataset_name: str) -> IndianDatasetSpec:
        if dataset_name not in self.dataset_specs:
            raise ValueError(f"Unknown dataset '{dataset_name}'.")
        return self.dataset_specs[dataset_name]

    @staticmethod
    def _normalize_label(
        raw_label: Any,
        label_map: Mapping[str, SentimentLabel],
    ) -> SentimentLabel | None:
        if isinstance(raw_label, SentimentLabel):
            return raw_label

        key = str(raw_label).strip().lower()
        if key in label_map:
            return label_map[key]

        if key in {"negative", "neutral", "positive"}:
            return SentimentLabel(key)
        return None


def tokenize_examples(
    examples: list[SentimentTrainingExample],
    *,
    tokenizer: BatchTokenizer | None = None,
    max_length: int = 64,
) -> dict[str, list[list[int]] | list[int]]:
    if not examples:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    active_tokenizer = tokenizer or SimpleWhitespaceTokenizer()
    encoded = active_tokenizer(
        [example.text for example in examples],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = [list(row) for row in encoded.get("input_ids", [])]
    attention_mask = [list(row) for row in encoded.get("attention_mask", [])]
    labels = [_label_to_id(example.label) for example in examples]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class FinBERTFineTunePipeline:
    def __init__(
        self,
        dataset_loader: IndianSentimentDatasetLoader | None = None,
        tokenizer: BatchTokenizer | None = None,
        *,
        max_length: int = 64,
        base_model_id: str = "ProsusAI/finbert",
    ):
        self.dataset_loader = dataset_loader or IndianSentimentDatasetLoader()
        self.tokenizer = tokenizer or SimpleWhitespaceTokenizer()
        self.max_length = max_length
        self.base_model_id = base_model_id

    def build_training_batch(
        self,
        raw_datasets: Mapping[str, Iterable[Mapping[str, Any]]],
    ) -> dict[str, Any]:
        merged_examples: list[SentimentTrainingExample] = []
        dataset_sizes: dict[str, int] = {}

        for dataset_name, rows in raw_datasets.items():
            examples = self.dataset_loader.load_examples(dataset_name, rows)
            merged_examples.extend(examples)
            dataset_sizes[dataset_name] = len(examples)

        tokenized = tokenize_examples(
            merged_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return {
            "base_model_id": self.base_model_id,
            "dataset_sizes": dataset_sizes,
            "total_examples": len(merged_examples),
            **tokenized,
        }
