"""
FinBERT Client — Calls the Dockerized FinBERT service.

Usage:
    from src.clients.finbert_client import FinBERTClient

    client = FinBERTClient()  # default: http://localhost:8001
    result = client.score("RBI cuts repo rate by 25 bps")
    print(result)  
    # {'sentiment_class': 'positive', 'sentiment_score': 0.87, 'confidence': 0.92, ...}

    # Batch scoring
    results = client.score_batch(["Market crashes 5%", "Nifty hits new high"])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8001"


@dataclass
class SentimentResult:
    sentiment_class: str
    sentiment_score: float
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


class FinBERTClient:
    """HTTP client for the Dockerized FinBERT microservice."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        """Check if service is healthy."""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def is_ready(self) -> bool:
        """Returns True if the service is up and model is loaded."""
        try:
            h = self.health()
            return h.get("ready", False)
        except Exception:
            return False

    def score(self, text: str) -> SentimentResult:
        """Score a single text."""
        resp = requests.post(
            f"{self.base_url}/score",
            json={"text": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return SentimentResult(
            sentiment_class=data["sentiment_class"],
            sentiment_score=data["sentiment_score"],
            confidence=data["confidence"],
            probabilities=data["probabilities"],
            latency_ms=data["latency_ms"],
        )

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Score up to 64 texts at once."""
        resp = requests.post(
            f"{self.base_url}/score/batch",
            json={"texts": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            SentimentResult(
                sentiment_class=r["sentiment_class"],
                sentiment_score=r["sentiment_score"],
                confidence=r["confidence"],
                probabilities=r["probabilities"],
                latency_ms=r["latency_ms"],
            )
            for r in data["results"]
        ]
