"""
FinBERT Sentiment Scoring Microservice

Exposes a REST API for sentiment classification using ProsusAI/finbert.
Designed to run inside Docker — model is pre-downloaded at build time.

Endpoints:
    POST /score          — score a single text
    POST /score/batch    — score multiple texts at once
    GET  /health         — health check
"""
from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("finbert-service")
logging.basicConfig(level=logging.INFO)

# ── Globals (loaded once at startup) ─────────────────────────────────────────

MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

tokenizer = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global tokenizer, model
    logger.info(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="FinBERT Sentiment Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response Schemas ───────────────────────────────────────────────


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)


class BatchScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class SentimentResult(BaseModel):
    sentiment_class: str
    sentiment_score: float      # positive - negative (range: -1 to +1)
    confidence: float           # max class probability
    probabilities: dict[str, float]  # all 3 class probs
    latency_ms: float


class BatchSentimentResult(BaseModel):
    results: list[SentimentResult]
    total_latency_ms: float


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ready": model is not None,
    }


@app.post("/score", response_model=SentimentResult)
async def score_text(req: ScoreRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    latency_ms = (time.time() - t0) * 1000

    return SentimentResult(
        sentiment_class=LABEL_MAP[pred_idx],
        sentiment_score=float(probs[0] - probs[1]),
        confidence=float(probs[pred_idx]),
        probabilities={
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2]),
        },
        latency_ms=round(latency_ms, 2),
    )


@app.post("/score/batch", response_model=BatchSentimentResult)
async def score_batch(req: BatchScoreRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    inputs = tokenizer(
        req.texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    all_probs = torch.softmax(outputs.logits, dim=1)
    results = []

    for i, probs in enumerate(all_probs):
        pred_idx = torch.argmax(probs).item()
        results.append(
            SentimentResult(
                sentiment_class=LABEL_MAP[pred_idx],
                sentiment_score=float(probs[0] - probs[1]),
                confidence=float(probs[pred_idx]),
                probabilities={
                    "positive": float(probs[0]),
                    "negative": float(probs[1]),
                    "neutral": float(probs[2]),
                },
                latency_ms=0,  # individual latency not meaningful in batch
            )
        )

    total_ms = (time.time() - t0) * 1000
    return BatchSentimentResult(
        results=results,
        total_latency_ms=round(total_ms, 2),
    )
