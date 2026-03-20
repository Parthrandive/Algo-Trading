from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.agents.sentiment.schemas import SentimentLabel


@dataclass(frozen=True)
class ModelSentimentOutput:
    label: SentimentLabel
    score: float
    confidence: float
    model_name: str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class KeywordSentimentModel:
    def __init__(
        self,
        *,
        positive_keywords: set[str] | None = None,
        negative_keywords: set[str] | None = None,
        model_name: str = "keyword_rule_v1",
    ):
        self.positive_keywords = positive_keywords or {
            "rally",
            "surge",
            "gain",
            "upbeat",
            "bullish",
            "buy",
            "beat",
            "growth",
            "strong",
            "upgrade",
        }
        self.negative_keywords = negative_keywords or {
            "fall",
            "drop",
            "slump",
            "weak",
            "bearish",
            "sell",
            "downgrade",
            "fraud",
            "loss",
            "panic",
        }
        self.model_name = model_name

    def predict(self, text: str) -> ModelSentimentOutput:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        positive_hits = sum(1 for token in tokens if token in self.positive_keywords)
        negative_hits = sum(1 for token in tokens if token in self.negative_keywords)
        net = positive_hits - negative_hits

        if net > 0:
            label = SentimentLabel.POSITIVE
        elif net < 0:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        raw_score = math.tanh(net / 3.0) if net else 0.0
        confidence = 0.55 + 0.1 * min(4, positive_hits + negative_hits)
        if label == SentimentLabel.NEUTRAL and positive_hits + negative_hits == 0:
            confidence = 0.5

        return ModelSentimentOutput(
            label=label,
            score=_clamp(raw_score, -1.0, 1.0),
            confidence=_clamp(confidence, 0.0, 1.0),
            model_name=self.model_name,
        )


class FinBERTSentimentModel:
    def __init__(
        self,
        *,
        model_id: str = "ProsusAI/finbert",
        classifier: Callable[..., Any] | None = None,
        fallback_model: KeywordSentimentModel | None = None,
    ):
        self.model_id = model_id
        self.classifier = classifier
        self.fallback_model = fallback_model or KeywordSentimentModel(model_name=f"{model_id}_fallback")

    @property
    def using_fallback(self) -> bool:
        return self.classifier is None

    @classmethod
    def bootstrap(
        cls,
        *,
        model_id: str = "ProsusAI/finbert",
        enable_hf_pipeline: bool = False,
        local_files_only: bool = True,
    ) -> "FinBERTSentimentModel":
        classifier: Callable[..., Any] | None = None
        if enable_hf_pipeline:
            try:
                from transformers import pipeline  # type: ignore

                classifier = pipeline(
                    task="text-classification",
                    model=model_id,
                    tokenizer=model_id,
                    truncation=True,
                    top_k=1,
                    local_files_only=local_files_only,
                )
            except Exception:
                classifier = None
        return cls(model_id=model_id, classifier=classifier)

    def predict(self, text: str) -> ModelSentimentOutput:
        if self.classifier is None:
            fallback = self.fallback_model.predict(text)
            return ModelSentimentOutput(
                label=fallback.label,
                score=fallback.score,
                confidence=fallback.confidence,
                model_name=fallback.model_name,
            )

        raw_result = self.classifier(text, truncation=True)
        normalized = self._extract_top_prediction(raw_result)
        label = self._normalize_label(str(normalized.get("label", "neutral")))
        confidence = float(normalized.get("score", 0.5))
        signed_score = confidence if label == SentimentLabel.POSITIVE else -confidence
        if label == SentimentLabel.NEUTRAL:
            signed_score = 0.0

        return ModelSentimentOutput(
            label=label,
            score=_clamp(signed_score, -1.0, 1.0),
            confidence=_clamp(confidence, 0.0, 1.0),
            model_name=self.model_id,
        )

    @staticmethod
    def _extract_top_prediction(raw_result: Any) -> dict[str, Any]:
        if isinstance(raw_result, list):
            if not raw_result:
                return {"label": "neutral", "score": 0.5}
            first = raw_result[0]
            if isinstance(first, list):
                if not first:
                    return {"label": "neutral", "score": 0.5}
                first = first[0]
            if isinstance(first, dict):
                return first
        if isinstance(raw_result, dict):
            return raw_result
        return {"label": "neutral", "score": 0.5}

    @staticmethod
    def _normalize_label(raw_label: str) -> SentimentLabel:
        normalized = raw_label.strip().lower()
        if "pos" in normalized:
            return SentimentLabel.POSITIVE
        if "neg" in normalized:
            return SentimentLabel.NEGATIVE
        if "neu" in normalized:
            return SentimentLabel.NEUTRAL
        return SentimentLabel.NEUTRAL
