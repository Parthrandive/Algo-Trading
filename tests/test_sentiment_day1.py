import json
from pathlib import Path

from src.agents.sentiment.datasets import (
    FinBERTFineTunePipeline,
    IndianSentimentDatasetLoader,
    SentimentTrainingExample,
    SimpleWhitespaceTokenizer,
    tokenize_examples,
)
from src.agents.sentiment.schemas import SentimentLabel
from src.agents.sentiment.sentiment_agent import SentimentAgent


def _runtime_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "sentiment_agent_runtime_v1.json"


def _load_runtime_config() -> dict:
    with _runtime_config_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_sentiment_runtime_config_day1_contract_freeze():
    config = _load_runtime_config()

    assert config["version"] == "sentiment-agent-runtime-v1"
    assert config["schema_version"] == "1.0"
    assert config["base_model"]["model_id"] == "ProsusAI/finbert"
    assert config["dual_speed"]["fast_lane"]["target_latency_ms"] <= 100

    assert set(config["datasets"]) == {
        "harixn/indian_news_sentiment",
        "SEntFiN",
    }
    assert set(config["cache_schema"]["required_fields"]) == {
        "score",
        "label",
        "generated_at_utc",
        "confidence",
        "ttl_seconds",
        "model_name",
        "source_id",
        "schema_version",
    }

    thresholds = config["precision_recall_thresholds"]
    for label in ("positive", "neutral", "negative"):
        assert thresholds[label]["precision_min"] > 0.0
        assert thresholds[label]["recall_min"] > 0.0


def test_indian_dataset_loader_normalizes_sentiment_labels():
    loader = IndianSentimentDatasetLoader()
    rows = [
        {"text": "Nifty rallies after RBI commentary", "sentiment": "positive"},
        {"text": "Banking index slips on weak guidance", "sentiment": "0"},
        {"text": "Market closes mixed and range-bound", "sentiment": "1"},
        {"text": "", "sentiment": "positive"},
    ]
    examples = loader.load_examples("harixn/indian_news_sentiment", rows)

    assert len(examples) == 3
    assert examples[0].label == SentimentLabel.POSITIVE
    assert examples[1].label == SentimentLabel.NEGATIVE
    assert examples[2].label == SentimentLabel.NEUTRAL


def test_tokenization_outputs_fixed_length_ids_and_masks():
    examples = [
        SentimentTrainingExample(
            text="Sensex surges as IT earnings beat estimates",
            label=SentimentLabel.POSITIVE,
        ),
        SentimentTrainingExample(
            text="Rupee weakens as oil spikes",
            label=SentimentLabel.NEGATIVE,
        ),
    ]
    encoded = tokenize_examples(
        examples,
        tokenizer=SimpleWhitespaceTokenizer(),
        max_length=12,
    )

    assert set(encoded.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(encoded["input_ids"]) == 2
    assert len(encoded["attention_mask"]) == 2
    assert len(encoded["input_ids"][0]) == 12
    assert len(encoded["attention_mask"][0]) == 12
    assert encoded["labels"] == [2, 0]


def test_finetune_pipeline_builds_combined_batch():
    pipeline = FinBERTFineTunePipeline(max_length=10)
    batch = pipeline.build_training_batch(
        {
            "harixn/indian_news_sentiment": [
                {"text": "Nifty closes higher", "sentiment": "positive"},
                {"text": "Small-cap index tumbles", "sentiment": "negative"},
            ],
            "SEntFiN": [
                {"text": "RBI pauses rates", "label": "0"},
            ],
        }
    )

    assert batch["base_model_id"] == "ProsusAI/finbert"
    assert batch["dataset_sizes"]["harixn/indian_news_sentiment"] == 2
    assert batch["dataset_sizes"]["SEntFiN"] == 1
    assert batch["total_examples"] == 3
    assert len(batch["input_ids"]) == 3
    assert len(batch["attention_mask"]) == 3
    assert len(batch["labels"]) == 3


def test_sentiment_agent_fast_lane_uses_cache_for_repeat_payloads():
    agent = SentimentAgent.from_default_components(_runtime_config_path())
    payload = {
        "source_id": "nse_news_001",
        "headline": "Nifty rally strengthens after inflation cools",
    }

    first = agent.score_textual_payload(payload, lane="fast")
    second = agent.score_textual_payload(payload, lane="fast")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.label == second.label
    assert first.text_hash == second.text_hash


def test_finbert_base_model_prediction_contract_on_indian_headline():
    agent = SentimentAgent.from_default_components(_runtime_config_path())

    prediction = agent.score(
        "RBI policy stance stays stable while markets remain cautious.",
        source_id="headline_sample_001",
        lane="slow",
    )

    assert prediction.label in {
        SentimentLabel.POSITIVE,
        SentimentLabel.NEGATIVE,
        SentimentLabel.NEUTRAL,
    }
    assert -1.0 <= prediction.score <= 1.0
    assert 0.0 <= prediction.confidence <= 1.0
    assert "ProsusAI/finbert" in prediction.model_name
