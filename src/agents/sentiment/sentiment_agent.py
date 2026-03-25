from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from sqlalchemy import select

from src.agents.sentiment.cache import InMemorySentimentCache, SentimentCacheBackend, build_cache_key
from src.agents.sentiment.cache_policy import evaluate_cache_policy, resolve_ttl_seconds
from src.agents.sentiment.fast_lane import FastLaneSentimentScorer
from src.agents.sentiment.models import FinBERTSentimentModel, KeywordSentimentModel, ModelSentimentOutput
from src.agents.sentiment.schemas import (
    CacheFreshnessState,
    DailySentimentAggregate,
    NightlySentimentBatchResult,
    SentimentLabel,
    SentimentLane,
    SentimentPrediction,
    SentimentQualityStatus,
)
from src.agents.sentiment.slow_lane import SlowLaneSentimentScorer
from src.agents.sentiment.text_utils import SentimentLanguageService, SentimentSafetyService
from src.db.models import SentimentScoreDB, TextItemDB
from src.db.phase2_recorder import Phase2Recorder

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "configs" / "sentiment_agent_runtime_v1.json"
DEFAULT_MODEL_CARD_PATH = ROOT_DIR / "data" / "models" / "sentiment" / "finbert_indian_v1" / "model_card.json"


class SentimentAgent:
    def __init__(
        self,
        *,
        fast_model: KeywordSentimentModel,
        slow_model: FinBERTSentimentModel,
        cache: SentimentCacheBackend | None = None,
        fast_ttl_seconds: int = 900,
        slow_ttl_seconds: int = 21600,
        fast_stale_after_seconds: int | None = None,
        slow_stale_after_seconds: int | None = None,
        stale_downweight_factor: float = 0.65,
        price_mismatch_threshold: float = 0.02,
        language_service: SentimentLanguageService | None = None,
        safety_service: SentimentSafetyService | None = None,
        phase2_recorder: Phase2Recorder | None = None,
        persist_predictions: bool = False,
        source_ttls: Mapping[str, int] | None = None,
        model_card_path: Path | None = None,
    ):
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.cache = cache or InMemorySentimentCache()
        self.fast_ttl_seconds = fast_ttl_seconds
        self.slow_ttl_seconds = slow_ttl_seconds
        self.fast_stale_after_seconds = fast_stale_after_seconds or max(1, int(fast_ttl_seconds * 0.6))
        self.slow_stale_after_seconds = slow_stale_after_seconds or max(1, int(slow_ttl_seconds * 0.6))
        self.stale_downweight_factor = min(max(stale_downweight_factor, 0.1), 1.0)
        self.price_mismatch_threshold = max(price_mismatch_threshold, 0.0)
        self.language_service = language_service or SentimentLanguageService()
        self.safety_service = safety_service or SentimentSafetyService()
        self.phase2_recorder = phase2_recorder
        self.persist_predictions = bool(persist_predictions and phase2_recorder is not None)
        self.source_ttls = {str(key): int(value) for key, value in dict(source_ttls or {}).items()}
        self.model_card_path = model_card_path or DEFAULT_MODEL_CARD_PATH
        self.fast_lane = FastLaneSentimentScorer(self, source_ttls=self.source_ttls)
        self.slow_lane = SlowLaneSentimentScorer(self, source_ttls=self.source_ttls)
        self._seen_source_fingerprints: set[str] = set()
        self._latest_aggregates: dict[str, DailySentimentAggregate] = {}
        self._model_card_registered = False

    @classmethod
    def from_default_components(
        cls,
        runtime_config_path: Path | None = None,
        *,
        database_url: str | None = None,
        phase2_recorder: Phase2Recorder | None = None,
        persist_predictions: bool = False,
    ) -> "SentimentAgent":
        config_path = runtime_config_path or DEFAULT_RUNTIME_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as handle:
            runtime_config = json.load(handle)

        dual_speed = runtime_config.get("dual_speed", {})
        fast_config = dual_speed.get("fast_lane", {})
        slow_config = dual_speed.get("slow_lane", {})
        cache_policy = runtime_config.get("cache_policy", {})
        robustness = runtime_config.get("robustness", {})
        model_config = runtime_config.get("base_model", {})

        keyword_lexicon = fast_config.get("keyword_lexicon", {})
        fast_model = KeywordSentimentModel(
            positive_keywords=set(keyword_lexicon.get("positive", [])),
            negative_keywords=set(keyword_lexicon.get("negative", [])),
            model_name=str(fast_config.get("model_name", "keyword_rule_v1")),
        )
        slow_model = FinBERTSentimentModel.bootstrap(
            model_id=str(model_config.get("model_id", "ProsusAI/finbert")),
            enable_hf_pipeline=bool(model_config.get("enable_hf_pipeline", False)),
            local_files_only=bool(model_config.get("local_files_only", True)),
            artifact_dir=model_config.get("fine_tuned_model_dir"),
        )

        recorder = phase2_recorder
        if recorder is None and persist_predictions:
            recorder = Phase2Recorder(database_url=database_url)

        model_card_dir = Path(str(model_config.get("fine_tuned_model_dir", DEFAULT_MODEL_CARD_PATH.parent)))
        return cls(
            fast_model=fast_model,
            slow_model=slow_model,
            cache=InMemorySentimentCache(),
            fast_ttl_seconds=int(fast_config.get("cache_ttl_seconds", 900)),
            slow_ttl_seconds=int(slow_config.get("cache_ttl_seconds", 21600)),
            fast_stale_after_seconds=int(cache_policy.get("fast_stale_after_seconds", 540)),
            slow_stale_after_seconds=int(cache_policy.get("slow_stale_after_seconds", 12960)),
            stale_downweight_factor=float(cache_policy.get("stale_downweight_factor", 0.65)),
            price_mismatch_threshold=float(robustness.get("price_mismatch_threshold", 0.02)),
            phase2_recorder=recorder,
            persist_predictions=persist_predictions,
            source_ttls=cache_policy.get("source_ttls_seconds", {}),
            model_card_path=model_card_dir / "model_card.json",
        )

    def score(
        self,
        text: str,
        *,
        source_id: str,
        lane: SentimentLane | str = SentimentLane.FAST,
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        active_lane = self._coerce_lane(lane)
        if active_lane == SentimentLane.FAST:
            return self.score_fast(text, source_id=source_id, as_of_utc=as_of_utc)
        return self.score_slow(text, source_id=source_id, as_of_utc=as_of_utc)

    def score_fast(
        self,
        text: str,
        *,
        source_id: str,
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        return self._score_lane(
            text=text,
            source_id=source_id,
            lane=SentimentLane.FAST,
            model=self.fast_model,
            ttl_seconds=self.fast_ttl_seconds,
            as_of_utc=as_of_utc,
        )

    def score_slow(
        self,
        text: str,
        *,
        source_id: str,
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        return self._score_lane(
            text=text,
            source_id=source_id,
            lane=SentimentLane.SLOW,
            model=self.slow_model,
            ttl_seconds=self.slow_ttl_seconds,
            as_of_utc=as_of_utc,
        )

    def score_textual_payload(
        self,
        payload: Mapping[str, Any],
        *,
        lane: SentimentLane | str = SentimentLane.FAST,
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        text_value = (
            payload.get("normalized_content")
            or payload.get("headline")
            or payload.get("content")
            or payload.get("text")
        )
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError("payload must include at least one non-empty text field: headline/content/text")

        source_id = str(payload.get("source_id", "unknown"))
        base_prediction = self.score(
            text_value,
            source_id=source_id,
            lane=lane,
            as_of_utc=as_of_utc,
        )
        return self._enrich_prediction(base_prediction=base_prediction, payload=payload, text=text_value)

    def score_textual_records(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        lane: SentimentLane | str = SentimentLane.FAST,
        dedupe: bool = True,
        as_of_utc: datetime | None = None,
    ) -> list[SentimentPrediction]:
        predictions: list[SentimentPrediction] = []
        seen_batch: set[str] = set()

        for payload in payloads:
            source_id = str(payload.get("source_id", "unknown"))
            text_value = (
                payload.get("normalized_content")
                or payload.get("headline")
                or payload.get("content")
                or payload.get("text")
            )
            if not isinstance(text_value, str) or not text_value.strip():
                continue

            _, text_hash = build_cache_key(lane=SentimentLane.FAST, text=text_value)
            batch_fingerprint = f"{source_id}:{text_hash}"
            if dedupe and batch_fingerprint in seen_batch:
                continue

            seen_batch.add(batch_fingerprint)
            predictions.append(
                self.score_textual_payload(
                    payload,
                    lane=lane,
                    as_of_utc=as_of_utc,
                )
            )

        return predictions
    def score_realtime(
        self,
        headline: str | Mapping[str, Any],
        *,
        source_id: str | None = None,
        symbol: str | None = None,
        source_type: str = "rss_feed",
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        payload = self._coerce_realtime_payload(
            headline,
            source_id=source_id,
            symbol=symbol,
            source_type=source_type,
            as_of_utc=as_of_utc,
        )
        prediction = self.fast_lane.score_headline(payload, as_of_utc=as_of_utc)
        self._persist_prediction(prediction)
        return prediction

    def run_nightly_batch(
        self,
        text_items: Iterable[Mapping[str, Any]] | None = None,
        *,
        as_of_utc: datetime | None = None,
        lookback_hours: int = 24,
        macro_by_symbol: Mapping[str, Mapping[str, float]] | None = None,
    ) -> NightlySentimentBatchResult:
        as_of = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        payloads = list(text_items) if text_items is not None else self._load_recent_text_items(as_of, lookback_hours)
        batch = self.slow_lane.run_nightly_batch(
            payloads,
            macro_by_symbol=macro_by_symbol,
            as_of_utc=as_of,
            lookback_hours=lookback_hours,
        )

        for prediction in batch.document_predictions:
            self._persist_prediction(prediction)

        for aggregate in batch.symbol_aggregates:
            self._persist_daily_aggregate(aggregate)
            if aggregate.symbol is not None:
                self._latest_aggregates[aggregate.symbol] = aggregate
        if batch.market_aggregate is not None:
            self._persist_daily_aggregate(batch.market_aggregate)
            self._latest_aggregates["MARKET"] = batch.market_aggregate
        return batch

    def get_z_t(
        self,
        symbol: str | None = None,
        *,
        as_of_utc: datetime | None = None,
    ) -> float:
        key = symbol or "MARKET"
        if key in self._latest_aggregates:
            return float(self._latest_aggregates[key].z_t)

        if self.phase2_recorder is None:
            return 0.0

        row = self._load_latest_sentiment_row(symbol=symbol, include_daily_agg=True, as_of_utc=as_of_utc)
        if row is None or row.z_t is None:
            return 0.0
        return float(row.z_t)

    def get_cached_sentiment(
        self,
        symbol: str | None,
        *,
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        as_of = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        row = self._load_latest_sentiment_row(symbol=symbol, include_daily_agg=False, as_of_utc=as_of)
        if row is None:
            return self._failure_prediction(
                source_id=symbol or "MARKET",
                text_hash=(symbol or "MARKET").lower(),
                lane=SentimentLane.FAST,
                decision_time=as_of,
                ttl_seconds=self.fast_ttl_seconds,
                model_name="cached_sentiment_missing",
                freshness_state=CacheFreshnessState.EXPIRED,
            )

        score_time = row.score_timestamp or row.timestamp
        if score_time.tzinfo is None:
            score_time = score_time.replace(tzinfo=UTC)
        else:
            score_time = score_time.astimezone(UTC)
        ttl_seconds = row.ttl_seconds or resolve_ttl_seconds(
            {"source_type": row.source_type, "lane": row.lane},
            fallback_seconds=self.fast_ttl_seconds if row.lane == SentimentLane.FAST.value else self.slow_ttl_seconds,
            source_ttls=self.source_ttls,
        )
        age_seconds = max(0.0, (as_of - score_time).total_seconds())
        decision = evaluate_cache_policy(age_seconds=age_seconds, ttl_seconds=ttl_seconds)
        base_prediction = self._row_to_prediction(row)
        base_prediction = base_prediction.model_copy(update={"freshness_state": decision.freshness_state})
        if decision.freshness_state == CacheFreshnessState.FRESH:
            return base_prediction
        if decision.freshness_state == CacheFreshnessState.STALE:
            return self._downweight_prediction(base_prediction)
        return self._failure_prediction(
            source_id=base_prediction.source_id,
            text_hash=base_prediction.text_hash,
            lane=base_prediction.lane,
            decision_time=as_of,
            ttl_seconds=ttl_seconds,
            model_name=base_prediction.model_name,
            freshness_state=CacheFreshnessState.EXPIRED,
        )

    def register_model_card(self, *, extra_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        now = datetime.now(UTC)
        training_meta = self._load_training_meta()
        performance = dict(training_meta.get("validation_metrics", {}))
        card = {
            "model_id": str(training_meta.get("model_id", "finbert_indian_v1")),
            "agent": "sentiment",
            "model_family": "dual_speed_sentiment",
            "version": str(training_meta.get("version", "1.0")),
            "created_at": now,
            "updated_at": now,
            "status": str(training_meta.get("status", "research_ready")),
            "performance": performance,
            "training_data_snapshot_hash": training_meta.get("training_data_snapshot_hash", "pending_external_dataset"),
            "code_hash": training_meta.get("code_hash", "workspace_current"),
            "feature_schema_version": "1.0",
            "hyperparameters": training_meta.get("hyperparameters", {}),
            "validation_metrics": performance,
            "baseline_comparison": training_meta.get("baseline_comparison", {}),
            "plan_version": "v1.3.7",
            "created_by": "Codex",
            "reviewed_by": "pending",
            "promotion_gate_checklist": {
                "timestamp_alignment": True,
                "cache_in_execution_path": False,
                "manipulation_detection": True,
                "thresholds_met": training_meta.get("threshold_status", {}),
            },
        }
        if extra_metadata:
            card.update(dict(extra_metadata))

        self.model_card_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_card_path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
        if self.phase2_recorder is not None:
            self.phase2_recorder.save_model_card(card)
        self._model_card_registered = True
        return card

    def compute_daily_z_t(
        self,
        predictions: Iterable[SentimentPrediction],
        *,
        macro_features: Mapping[str, float] | None = None,
        as_of_utc: datetime | None = None,
        symbol: str | None = None,
    ) -> DailySentimentAggregate:
        decision_time = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        prediction_list = list(predictions)
        sample_size = len(prediction_list)

        weighted_numerator = 0.0
        confidence_total = 0.0
        quality_warn = False
        for prediction in prediction_list:
            weighted_numerator += prediction.score * prediction.confidence
            confidence_total += prediction.confidence
            if prediction.quality_status in {SentimentQualityStatus.WARN, SentimentQualityStatus.FAIL}:
                quality_warn = True

        weighted_sentiment_score = weighted_numerator / confidence_total if confidence_total > 0 else 0.0
        sentiment_confidence = confidence_total / sample_size if sample_size > 0 else 0.0

        macro_adjustment = self._compute_macro_adjustment(macro_features)
        z_t = self._clamp((0.8 * weighted_sentiment_score) + (0.2 * macro_adjustment), -1.0, 1.0)

        quality_status = SentimentQualityStatus.PASS
        if sample_size == 0:
            quality_status = SentimentQualityStatus.FAIL
        elif quality_warn or sentiment_confidence < 0.45:
            quality_status = SentimentQualityStatus.WARN

        return DailySentimentAggregate(
            generated_at_utc=decision_time,
            symbol=symbol,
            sample_size=sample_size,
            weighted_sentiment_score=self._clamp(weighted_sentiment_score, -1.0, 1.0),
            macro_adjustment=self._clamp(macro_adjustment, -1.0, 1.0),
            z_t=z_t,
            sentiment_confidence=self._clamp(sentiment_confidence, 0.0, 1.0),
            quality_status=quality_status,
            provenance={
                "component": "sentiment_agent",
                "macro_feature_count": 0 if macro_features is None else len(macro_features),
            },
        )
    def _score_lane(
        self,
        *,
        text: str,
        source_id: str,
        lane: SentimentLane,
        model: KeywordSentimentModel | FinBERTSentimentModel,
        ttl_seconds: int,
        as_of_utc: datetime | None,
    ) -> SentimentPrediction:
        decision_time = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        cache_key, text_hash = build_cache_key(lane=lane, text=text)
        stale_after_seconds = self.fast_stale_after_seconds if lane == SentimentLane.FAST else self.slow_stale_after_seconds
        freshness_on_miss = CacheFreshnessState.MISS

        try:
            cache_entry = self.cache.get(cache_key, as_of_utc=decision_time, include_expired=True)
        except Exception:
            return self._failure_prediction(
                source_id=source_id,
                text_hash=text_hash,
                lane=lane,
                decision_time=decision_time,
                ttl_seconds=ttl_seconds,
                model_name=f"{lane.value}_cache_failure",
                freshness_state=CacheFreshnessState.ERROR,
            )

        if cache_entry is not None:
            age_seconds = max(0.0, (decision_time - cache_entry.generated_at_utc).total_seconds())
            if age_seconds <= stale_after_seconds:
                return self._prediction_from_cache_entry(
                    source_id=source_id,
                    cache_entry=cache_entry,
                    freshness_state=CacheFreshnessState.FRESH,
                    cache_hit=True,
                )

            if age_seconds <= cache_entry.ttl_seconds:
                stale_prediction = self._prediction_from_cache_entry(
                    source_id=source_id,
                    cache_entry=cache_entry,
                    freshness_state=CacheFreshnessState.STALE,
                    cache_hit=True,
                )
                return self._downweight_prediction(stale_prediction)

            freshness_on_miss = CacheFreshnessState.EXPIRED

        try:
            model_output = model.predict(text)
        except Exception:
            return self._failure_prediction(
                source_id=source_id,
                text_hash=text_hash,
                lane=lane,
                decision_time=decision_time,
                ttl_seconds=ttl_seconds,
                model_name=f"{lane.value}_model_failure",
                freshness_state=freshness_on_miss,
            )

        new_entry = self._build_cache_entry(
            source_id=source_id,
            text_hash=text_hash,
            cache_key=cache_key,
            lane=lane,
            model_output=model_output,
            generated_at_utc=decision_time,
            ttl_seconds=ttl_seconds,
        )

        reduced_risk_mode = False
        fallback_mode: str | None = None
        try:
            self.cache.set(cache_key, new_entry)
        except Exception:
            reduced_risk_mode = True
            fallback_mode = "technical_only_reduced_risk"

        return SentimentPrediction(
            source_id=new_entry.source_id,
            text_hash=new_entry.text_hash,
            lane=new_entry.lane,
            label=new_entry.label,
            score=new_entry.score,
            confidence=new_entry.confidence,
            generated_at_utc=new_entry.generated_at_utc,
            score_timestamp_utc=new_entry.generated_at_utc,
            ttl_seconds=new_entry.ttl_seconds,
            model_name=new_entry.model_name,
            cache_hit=False,
            freshness_state=freshness_on_miss,
            reduced_risk_mode=reduced_risk_mode,
            fallback_mode=fallback_mode,
            provenance={"cache_write_ok": not reduced_risk_mode},
        )

    @staticmethod
    def _build_cache_entry(
        *,
        source_id: str,
        text_hash: str,
        cache_key: str,
        lane: SentimentLane,
        model_output: ModelSentimentOutput,
        generated_at_utc: datetime,
        ttl_seconds: int,
    ):
        from src.agents.sentiment.schemas import SentimentCacheEntry

        return SentimentCacheEntry(
            cache_key=cache_key,
            source_id=source_id,
            text_hash=text_hash,
            lane=lane,
            label=model_output.label,
            score=model_output.score,
            confidence=model_output.confidence,
            generated_at_utc=generated_at_utc,
            ttl_seconds=ttl_seconds,
            model_name=model_output.model_name,
        )

    @staticmethod
    def _coerce_lane(lane: SentimentLane | str) -> SentimentLane:
        if isinstance(lane, SentimentLane):
            return lane
        return SentimentLane(str(lane).strip().lower())

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _prediction_from_cache_entry(
        self,
        *,
        source_id: str,
        cache_entry,
        freshness_state: CacheFreshnessState,
        cache_hit: bool,
    ) -> SentimentPrediction:
        return SentimentPrediction(
            source_id=source_id,
            text_hash=cache_entry.text_hash,
            lane=cache_entry.lane,
            label=cache_entry.label,
            score=cache_entry.score,
            confidence=cache_entry.confidence,
            generated_at_utc=cache_entry.generated_at_utc,
            score_timestamp_utc=cache_entry.generated_at_utc,
            ttl_seconds=cache_entry.ttl_seconds,
            model_name=cache_entry.model_name,
            cache_hit=cache_hit,
            freshness_state=freshness_state,
        )

    def _downweight_prediction(self, prediction: SentimentPrediction) -> SentimentPrediction:
        return prediction.model_copy(
            update={
                "score": self._clamp(prediction.score * self.stale_downweight_factor, -1.0, 1.0),
                "confidence": self._clamp(prediction.confidence * self.stale_downweight_factor, 0.0, 1.0),
                "downweighted": True,
            }
        )

    def _failure_prediction(
        self,
        *,
        source_id: str,
        text_hash: str,
        lane: SentimentLane,
        decision_time: datetime,
        ttl_seconds: int,
        model_name: str,
        freshness_state: CacheFreshnessState,
    ) -> SentimentPrediction:
        return SentimentPrediction(
            source_id=source_id,
            text_hash=text_hash,
            lane=lane,
            label=SentimentLabel.NEUTRAL,
            score=0.0,
            confidence=0.0,
            generated_at_utc=decision_time,
            score_timestamp_utc=decision_time,
            ttl_seconds=ttl_seconds,
            model_name=model_name,
            cache_hit=False,
            freshness_state=freshness_state,
            quality_status=SentimentQualityStatus.FAIL,
            reduced_risk_mode=True,
            fallback_mode="technical_only_reduced_risk",
            provenance={"failure_mode": "cache_or_model_failure"},
        )

    def _enrich_prediction(
        self,
        *,
        base_prediction: SentimentPrediction,
        payload: Mapping[str, Any],
        text: str,
    ) -> SentimentPrediction:
        language = str(payload.get("language") or self.language_service.detect_language(text)).strip().lower()
        normalized_text = str(payload.get("normalized_content") or "").strip()
        if not normalized_text and language in {"hi", "code_mixed"}:
            normalized_text = self.language_service.normalize_hinglish(text)

        transliterated_text = ""
        if language in {"hi", "code_mixed"}:
            transliterated_text = self.language_service.transliterate_to_latin(text)

        scam_flags, scam_risk = self.safety_service.check_for_scams(
            text,
            normalized_text=normalized_text or None,
            transliterated_text=transliterated_text or None,
        )
        adversarial_flags, adversarial_risk = self.safety_service.check_for_adversarial_patterns(text)

        flags: list[str] = []
        if language == "code_mixed":
            flags.append("code_mixed_detected")
        if language == "hi":
            flags.append("hindi_detected")
        flags.extend(scam_flags)
        flags.extend(adversarial_flags)

        payload_risk = payload.get("manipulation_risk_score", 0.0)
        try:
            payload_risk_float = float(payload_risk)
        except (TypeError, ValueError):
            payload_risk_float = 0.0

        manipulation_risk = self._clamp(payload_risk_float + scam_risk + adversarial_risk, 0.0, 1.0)

        duplicate = self._is_duplicate(base_prediction.source_id, base_prediction.text_hash)
        if duplicate:
            flags.append("source_duplicate")

        quality_status = SentimentQualityStatus.PASS
        if manipulation_risk >= 0.7:
            quality_status = SentimentQualityStatus.FAIL
        elif manipulation_risk >= 0.35 or duplicate:
            quality_status = SentimentQualityStatus.WARN

        effective_prediction = base_prediction
        if quality_status == SentimentQualityStatus.FAIL:
            effective_prediction = effective_prediction.model_copy(
                update={
                    "label": SentimentLabel.NEUTRAL,
                    "score": 0.0,
                    "confidence": min(effective_prediction.confidence, 0.5),
                    "downweighted": True,
                }
            )

        effective_prediction = self._apply_price_mismatch_circuit_breaker(effective_prediction, payload, flags)
        headline_timestamp = payload.get("timestamp")
        if isinstance(headline_timestamp, datetime) and headline_timestamp.tzinfo is not None:
            headline_timestamp = headline_timestamp.astimezone(UTC)
        else:
            headline_timestamp = None

        ttl_seconds = resolve_ttl_seconds(
            payload,
            fallback_seconds=self.fast_ttl_seconds if effective_prediction.lane == SentimentLane.FAST else self.slow_ttl_seconds,
            source_ttls=self.source_ttls,
        )

        return effective_prediction.model_copy(
            update={
                "symbol": payload.get("symbol"),
                "source_type": payload.get("source_type"),
                "headline_timestamp_utc": headline_timestamp,
                "score_timestamp_utc": effective_prediction.generated_at_utc,
                "ttl_seconds": ttl_seconds,
                "quality_status": quality_status,
                "manipulation_risk_score": manipulation_risk,
                "manipulation_flags": tuple(dict.fromkeys(flags)),
                "provenance": {
                    "language": language,
                    "normalized_content_used": bool(normalized_text),
                    "transliterated_content_used": bool(transliterated_text),
                },
            }
        )
    def _apply_price_mismatch_circuit_breaker(
        self,
        prediction: SentimentPrediction,
        payload: Mapping[str, Any],
        flags: list[str],
    ) -> SentimentPrediction:
        raw_price_return = payload.get("price_return")
        if raw_price_return is None:
            return prediction

        try:
            price_return = float(raw_price_return)
        except (TypeError, ValueError):
            return prediction

        score = prediction.score
        mismatch = (score >= 0.35 and price_return <= -self.price_mismatch_threshold) or (
            score <= -0.35 and price_return >= self.price_mismatch_threshold
        )
        if not mismatch:
            return prediction

        flags.append("sentiment_price_mismatch")
        return prediction.model_copy(
            update={
                "label": SentimentLabel.NEUTRAL,
                "score": 0.0,
                "confidence": min(prediction.confidence, 0.55),
                "downweighted": True,
                "quality_status": SentimentQualityStatus.WARN,
            }
        )

    def _is_duplicate(self, source_id: str, text_hash: str) -> bool:
        fingerprint = f"{source_id}:{text_hash}"
        if fingerprint in self._seen_source_fingerprints:
            return True
        self._seen_source_fingerprints.add(fingerprint)
        return False

    def _compute_macro_adjustment(self, macro_features: Mapping[str, float] | None) -> float:
        if not macro_features:
            return 0.0

        values: list[float] = []
        for value in macro_features.values():
            try:
                values.append(self._clamp(float(value), -1.0, 1.0))
            except (TypeError, ValueError):
                continue
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _coerce_realtime_payload(
        self,
        headline: str | Mapping[str, Any],
        *,
        source_id: str | None,
        symbol: str | None,
        source_type: str,
        as_of_utc: datetime | None,
    ) -> Mapping[str, Any]:
        if isinstance(headline, Mapping):
            payload = dict(headline)
            payload.setdefault("source_id", source_id or "realtime_headline")
            payload.setdefault("symbol", symbol)
            payload.setdefault("source_type", source_type)
            payload.setdefault("timestamp", (as_of_utc or datetime.now(UTC)).astimezone(UTC))
            return payload

        return {
            "source_id": source_id or "realtime_headline",
            "headline": str(headline),
            "symbol": symbol,
            "source_type": source_type,
            "timestamp": (as_of_utc or datetime.now(UTC)).astimezone(UTC),
        }

    def _load_recent_text_items(self, as_of_utc: datetime, lookback_hours: int) -> list[dict[str, Any]]:
        if self.phase2_recorder is None:
            return []

        start = as_of_utc - timedelta(hours=lookback_hours)
        with self.phase2_recorder.Session() as session:
            rows = session.execute(
                select(TextItemDB)
                .where(TextItemDB.timestamp >= start, TextItemDB.timestamp <= as_of_utc)
                .order_by(TextItemDB.timestamp.asc())
            ).scalars().all()
        return [self._text_item_to_payload(row) for row in rows]

    @staticmethod
    def _text_item_to_payload(row: TextItemDB) -> dict[str, Any]:
        return {
            "source_id": row.source_id,
            "timestamp": row.timestamp,
            "headline": row.headline or row.content[:160],
            "content": row.content,
            "symbol": row.symbol,
            "source_type": row.source_type,
            "item_type": row.item_type,
            "language": row.language,
            "publisher": row.publisher,
            "platform": row.platform,
            "author": row.author,
        }

    def _persist_prediction(self, prediction: SentimentPrediction) -> None:
        if not self.persist_predictions or self.phase2_recorder is None:
            return
        self._ensure_model_card_registered()
        self.phase2_recorder.save_sentiment_score(self._prediction_to_row(prediction))

    def _persist_daily_aggregate(self, aggregate: DailySentimentAggregate) -> None:
        if not self.persist_predictions or self.phase2_recorder is None:
            return
        self._ensure_model_card_registered()
        self.phase2_recorder.save_sentiment_score(self._aggregate_to_row(aggregate))

    def _ensure_model_card_registered(self) -> None:
        if not self._model_card_registered:
            self.register_model_card()

    def _prediction_to_row(self, prediction: SentimentPrediction) -> dict[str, Any]:
        timestamp = prediction.score_timestamp_utc or prediction.generated_at_utc
        return {
            "symbol": prediction.symbol,
            "timestamp": timestamp,
            "lane": prediction.lane.value,
            "source_id": prediction.source_id,
            "source_type": prediction.source_type,
            "sentiment_class": prediction.label.value,
            "sentiment_score": prediction.score,
            "z_t": None,
            "confidence": prediction.confidence,
            "source_count": 1,
            "ttl_seconds": prediction.ttl_seconds,
            "freshness_flag": prediction.freshness_state.value,
            "headline_timestamp": prediction.headline_timestamp_utc,
            "score_timestamp": timestamp,
            "quality_status": prediction.quality_status.value,
            "metadata": {
                "cache_hit": prediction.cache_hit,
                "downweighted": prediction.downweighted,
                "manipulation_flags": list(prediction.manipulation_flags),
                "manipulation_risk_score": prediction.manipulation_risk_score,
                "reduced_risk_mode": prediction.reduced_risk_mode,
                "fallback_mode": prediction.fallback_mode,
                "provenance": prediction.provenance,
                "latency_ms": prediction.latency_ms,
            },
            "model_id": prediction.model_name,
            "schema_version": prediction.schema_version,
        }

    def _aggregate_to_row(self, aggregate: DailySentimentAggregate) -> dict[str, Any]:
        return {
            "symbol": aggregate.symbol,
            "timestamp": aggregate.generated_at_utc,
            "lane": aggregate.lane.value,
            "source_id": aggregate.symbol or "MARKET",
            "source_type": "aggregate",
            "sentiment_class": self._score_to_label(aggregate.weighted_sentiment_score).value,
            "sentiment_score": aggregate.weighted_sentiment_score,
            "z_t": aggregate.z_t,
            "confidence": aggregate.sentiment_confidence,
            "source_count": aggregate.sample_size,
            "ttl_seconds": 24 * 60 * 60,
            "freshness_flag": CacheFreshnessState.FRESH.value,
            "headline_timestamp": None,
            "score_timestamp": aggregate.generated_at_utc,
            "quality_status": aggregate.quality_status.value,
            "metadata": {"provenance": aggregate.provenance},
            "model_id": "sentiment_daily_agg_v1",
            "schema_version": "1.0",
        }

    @staticmethod
    def _score_to_label(score: float) -> SentimentLabel:
        if score >= 0.1:
            return SentimentLabel.POSITIVE
        if score <= -0.1:
            return SentimentLabel.NEGATIVE
        return SentimentLabel.NEUTRAL

    def _load_latest_sentiment_row(
        self,
        *,
        symbol: str | None,
        include_daily_agg: bool,
        as_of_utc: datetime | None,
    ) -> SentimentScoreDB | None:
        if self.phase2_recorder is None:
            return None

        cutoff = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        with self.phase2_recorder.Session() as session:
            stmt = select(SentimentScoreDB).where(SentimentScoreDB.timestamp <= cutoff)
            if symbol is None:
                stmt = stmt.where(SentimentScoreDB.symbol.is_(None))
            else:
                stmt = stmt.where(SentimentScoreDB.symbol == symbol)
            if not include_daily_agg:
                stmt = stmt.where(SentimentScoreDB.lane != SentimentLane.DAILY_AGG.value)
            stmt = stmt.order_by(SentimentScoreDB.timestamp.desc(), SentimentScoreDB.id.desc())
            return session.execute(stmt).scalars().first()

    def _row_to_prediction(self, row: SentimentScoreDB) -> SentimentPrediction:
        timestamp = row.score_timestamp or row.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)
        headline_timestamp = row.headline_timestamp
        if headline_timestamp is not None:
            if headline_timestamp.tzinfo is None:
                headline_timestamp = headline_timestamp.replace(tzinfo=UTC)
            else:
                headline_timestamp = headline_timestamp.astimezone(UTC)
        lane = self._coerce_lane(row.lane)
        text_hash = build_cache_key(lane=lane, text=f"{row.source_id}:{row.timestamp.isoformat()}")[1]
        return SentimentPrediction(
            source_id=row.source_id or (row.symbol or "MARKET"),
            text_hash=text_hash,
            lane=lane,
            label=SentimentLabel(str(row.sentiment_class).lower()),
            symbol=row.symbol,
            source_type=row.source_type,
            score=float(row.sentiment_score),
            confidence=float(row.confidence),
            generated_at_utc=timestamp,
            headline_timestamp_utc=headline_timestamp,
            score_timestamp_utc=timestamp,
            ttl_seconds=int(row.ttl_seconds or self.fast_ttl_seconds),
            model_name=row.model_id,
            freshness_state=CacheFreshnessState(str(row.freshness_flag or CacheFreshnessState.MISS.value)),
            quality_status=SentimentQualityStatus(str(row.quality_status or SentimentQualityStatus.PASS.value)),
        )

    def _load_training_meta(self) -> dict[str, Any]:
        training_meta_path = self.model_card_path.parent / "training_meta.json"
        if training_meta_path.exists():
            try:
                return json.loads(training_meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}
