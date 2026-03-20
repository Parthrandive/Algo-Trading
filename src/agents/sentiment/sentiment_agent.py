from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.agents.sentiment.cache import InMemorySentimentCache, SentimentCacheBackend, build_cache_key
from src.agents.sentiment.models import FinBERTSentimentModel, KeywordSentimentModel, ModelSentimentOutput
from src.agents.sentiment.schemas import (
    CacheFreshnessState,
    DailySentimentAggregate,
    SentimentCacheEntry,
    SentimentLabel,
    SentimentLane,
    SentimentPrediction,
    SentimentQualityStatus,
)
from src.agents.sentiment.text_utils import SentimentLanguageService, SentimentSafetyService

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "configs" / "sentiment_agent_runtime_v1.json"


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
        self._seen_source_fingerprints: set[str] = set()

    @classmethod
    def from_default_components(cls, runtime_config_path: Path | None = None) -> "SentimentAgent":
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
        )

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

    def compute_daily_z_t(
        self,
        predictions: Iterable[SentimentPrediction],
        *,
        macro_features: Mapping[str, float] | None = None,
        as_of_utc: datetime | None = None,
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
    ) -> SentimentCacheEntry:
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
        cache_entry: SentimentCacheEntry,
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

        return effective_prediction.model_copy(
            update={
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
