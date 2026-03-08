from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ValidationError

from src.agents.textual.adapters import RawTextRecord
from src.agents.textual.services.safety_service import SafetyService
from src.schemas.text_sidecar import ComplianceStatus, TextSidecarMetadata
from src.utils.schema_registry import SchemaRegistry


@dataclass(frozen=True)
class ComplianceDecision:
    status: ComplianceStatus
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.status == ComplianceStatus.ALLOW


class TextualValidator:
    _BASE_CANONICAL_FIELDS = {
        "source_id",
        "timestamp",
        "content",
        "source_type",
        "url",
        "author",
        "language",
        "ingestion_timestamp_utc",
        "ingestion_timestamp_ist",
        "schema_version",
        "quality_status",
    }
    _RECORD_TYPE_CANONICAL_FIELDS = {
        "news_article": {"headline", "publisher"},
        "social_post": {"platform", "likes", "shares"},
        "earnings_transcript": {"symbol", "quarter", "year"},
    }
    _QUALITY_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}

    def __init__(self, runtime_config: Mapping[str, Any]):
        self.runtime_config = dict(runtime_config)
        self._allowlist = {
            str(entry["source_name"]): dict(entry)
            for entry in self.runtime_config.get("source_allowlist", [])
        }
        self._canonical_schema_keys = dict(self.runtime_config.get("canonical_schema_keys", {}))
        self._default_ttl_seconds = dict(self.runtime_config.get("default_ttl_seconds", {}))
        self._global_rules = dict(self.runtime_config.get("global_compliance_rules", {}))

        # Day 3: X-specific templates and filters.
        x_templates = self.runtime_config.get("x_query_templates", {})
        self._india_relevance_keywords = {
            "nifty",
            "sensex",
            "india",
            "rbi",
            "mpc",
            "repo",
            "nse",
            "infy",
            "tcs",
            "hdfc",
            "reliance",
        }
        self._negative_filters = set(x_templates.get("negative_filters", []))

        # Day 4/5: Safety and PDF scoring controls.
        self.safety_service = SafetyService()
        pdf_quality_config = dict(self.runtime_config.get("pdf_quality", {}))
        self._pdf_warn_below = min(max(float(pdf_quality_config.get("warn_below", 0.8)), 0.0), 1.0)
        self._pdf_fail_below = min(max(float(pdf_quality_config.get("fail_below", 0.6)), 0.0), 1.0)
        if self._pdf_fail_below > self._pdf_warn_below:
            self._pdf_fail_below, self._pdf_warn_below = self._pdf_warn_below, self._pdf_fail_below

<<<<<<< HEAD
        quality_config = dict(self.runtime_config.get("quality_controls", {}))
        self._min_content_chars = max(int(quality_config.get("min_content_chars", 20)), 0)
        self._max_content_chars = max(
            int(quality_config.get("max_content_chars", 12000)),
            self._min_content_chars + 1,
        )
        self._max_future_skew_seconds = max(int(quality_config.get("max_future_skew_seconds", 300)), 0)
=======
        quality_controls = dict(self.runtime_config.get("quality_controls", {}))
        self._min_content_chars = max(int(quality_controls.get("min_content_chars", 20)), 0)
        self._max_content_chars = max(int(quality_controls.get("max_content_chars", 30000)), self._min_content_chars + 1)
        self._max_future_skew_seconds = max(int(quality_controls.get("max_future_skew_seconds", 300)), 0)
        freshness_windows = quality_controls.get("freshness_windows_seconds", {})
        if isinstance(freshness_windows, dict):
            self._freshness_windows_seconds = {str(k): max(int(v), 0) for k, v in freshness_windows.items()}
        else:
            self._freshness_windows_seconds = {}
>>>>>>> 701ccfb8293a2001f6b46632488e94f99447ad31

    @property
    def pdf_warn_below(self) -> float:
        return self._pdf_warn_below

    @property
    def pdf_fail_below(self) -> float:
        return self._pdf_fail_below

    @classmethod
    def from_config_path(cls, config_path: Path) -> "TextualValidator":
        with config_path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def resolve_schema_key(self, record_type: str) -> str:
        if record_type not in self._canonical_schema_keys:
            raise ValueError(f"No canonical schema mapping configured for record_type={record_type}")
        return str(self._canonical_schema_keys[record_type])

    def default_ttl_seconds(self, record_type: str) -> int:
        return int(self._default_ttl_seconds.get(record_type, 0))

    def validate_canonical(self, record_type: str, payload: Mapping[str, Any]) -> BaseModel:
        schema_key = self.resolve_schema_key(record_type)
        return SchemaRegistry.validate(schema_key, dict(payload))

    def apply_quality_controls(
        self,
        record: RawTextRecord,
        payload: Mapping[str, Any],
        *,
        run_timestamp: datetime,
        seen_source_keys: set[tuple[str, str]],
        seen_content_fingerprints: set[str],
    ) -> dict[str, Any]:
        payload_dict = dict(payload)
        quality_flags = self._extract_payload_quality_flags(payload_dict)
        quality_status = self._coerce_quality_status(payload_dict.get("quality_status"))
        risk_increment = 0.0

        content = record.content.strip()
        if not content:
            quality_flags.append("missing_content")
            quality_status = self._upgrade_quality_status(quality_status, "fail")
        elif len(content) < self._min_content_chars:
            quality_flags.append("content_too_short")
            quality_status = self._upgrade_quality_status(quality_status, "warn")

        if len(content) > self._max_content_chars:
            quality_flags.append("content_too_long")
            quality_status = self._upgrade_quality_status(quality_status, "warn")

        for required_field in self._required_fields_for_record_type(record.record_type):
            value = payload_dict.get(required_field)
            if isinstance(value, str):
                value = value.strip()
            if value in (None, ""):
                quality_flags.append(f"missing_{required_field}")
                quality_status = self._upgrade_quality_status(quality_status, "fail")

        if not isinstance(record.timestamp, datetime) or record.timestamp.tzinfo is None:
            quality_flags.append("malformed_timestamp")
            quality_status = self._upgrade_quality_status(quality_status, "fail")
        else:
            run_utc = run_timestamp.astimezone(UTC)
            event_utc = record.timestamp.astimezone(UTC)
            if event_utc > run_utc + timedelta(seconds=self._max_future_skew_seconds):
                quality_flags.append("future_timestamp")
                quality_status = self._upgrade_quality_status(quality_status, "warn")

        source_key = (record.source_type.value, record.source_id)
        if source_key in seen_source_keys:
            quality_flags.append("duplicate_source_id")
            quality_status = self._upgrade_quality_status(quality_status, "warn")
            risk_increment += 0.1
        else:
            seen_source_keys.add(source_key)

        content_fingerprint = self._fingerprint_content(content)
        if content_fingerprint:
            if content_fingerprint in seen_content_fingerprints:
                quality_flags.append("duplicate_content")
                quality_status = self._upgrade_quality_status(quality_status, "warn")
                risk_increment += 0.1
            else:
                seen_content_fingerprints.add(content_fingerprint)

        adversarial_flags, adversarial_risk = self.safety_service.check_for_adversarial_patterns(record.content)
        if adversarial_flags:
            quality_flags.extend(adversarial_flags)
            quality_status = self._upgrade_quality_status(quality_status, "warn")
            risk_increment += adversarial_risk

        base_risk = self._coerce_float(payload_dict.get("manipulation_risk_score"), default=0.0)
        payload_dict["manipulation_risk_score"] = self._clamp01(base_risk + risk_increment)
        payload_dict["quality_flags"] = quality_flags
        payload_dict["quality_status"] = quality_status
        return payload_dict

    def validate_record(
        self,
        record: RawTextRecord,
        payload: Mapping[str, Any],
    ) -> tuple[BaseModel | None, TextSidecarMetadata]:
        payload_dict = dict(payload)
        payload_flags = self._extract_payload_quality_flags(payload_dict)

        scam_flags, scam_risk_increment = self.safety_service.check_for_scams(
            record.content,
            normalized_text=payload_dict.get("normalized_content")
            if isinstance(payload_dict.get("normalized_content"), str)
            else None,
            transliterated_text=payload_dict.get("transliterated_content")
            if isinstance(payload_dict.get("transliterated_content"), str)
            else None,
        )
        pdf_quality_flags = self._pdf_quality_flags_for_payload(payload_dict)
        extraction_score = self._extract_pdf_quality_score(payload_dict)
        decision = self.evaluate_compliance(record, payload_dict)

        if not decision.allowed:
<<<<<<< HEAD
            reject_flags = ["compliance_reject"]
            reject_flags.extend(payload_flags)
            reject_flags.extend(pdf_quality_flags)
            reject_flags.extend(scam_flags)
=======
            reject_quality_flags = ["compliance_reject"]
            reject_quality_flags.extend(self._extract_payload_quality_flags(payload_dict))
            reject_quality_flags.extend(quality_gate_flags)
            reject_quality_flags.extend(pdf_quality_flags)
            reject_quality_flags.extend(scam_flags)
>>>>>>> 701ccfb8293a2001f6b46632488e94f99447ad31
            return None, self._build_reject_sidecar(
                record=record,
                payload=payload_dict,
                reason=str(decision.reason),
                quality_flags=reject_quality_flags,
                risk_increment=scam_risk_increment,
            )

        if self._has_duplicate_flag(payload_flags):
            duplicate_flags = list(payload_flags)
            duplicate_flags.extend(quality_gate_flags)
            duplicate_flags.extend(pdf_quality_flags)
            duplicate_flags.extend(scam_flags)
            return None, self._build_reject_sidecar(
                record=record,
                payload=payload_dict,
                reason="duplicate_record",
                quality_flags=duplicate_flags,
                risk_increment=scam_risk_increment,
            )

        try:
            canonical_record = self.validate_canonical(
                record.record_type,
                self._canonical_payload_only(record.record_type, payload_dict),
            )
        except (ValidationError, ValueError, TypeError):
            invalid_flags = list(payload_flags)
            invalid_flags.extend(quality_gate_flags)
            invalid_flags.extend(pdf_quality_flags)
            invalid_flags.extend(scam_flags)
            invalid_flags.append("canonical_validation_error")
            return None, self._build_reject_sidecar(
                record=record,
                payload=payload_dict,
                reason="canonical_validation_error",
                quality_flags=invalid_flags,
                risk_increment=scam_risk_increment,
            )

        sidecar_quality_flags = self._extract_quality_flags(payload_dict, canonical_record)
        sidecar_quality_flags.extend(pdf_quality_flags)
        sidecar_quality_flags.extend(scam_flags)

        ingestion_timestamp_utc = getattr(canonical_record, "ingestion_timestamp_utc")
        ttl_seconds = self.default_ttl_seconds(record.record_type)

        # Day 3: Reliability and confidence adjustments.
        base_confidence = self._coerce_float(payload_dict.get("confidence"), default=0.5)
        adjusted_confidence = self._calculate_adjusted_confidence(record, payload_dict, base_confidence)

        # Day 4: PDF extraction quality should downgrade confidence on weak extraction.
        if extraction_score is not None:
            adjusted_confidence *= extraction_score
            if extraction_score < self._pdf_fail_below:
                adjusted_confidence *= 0.8

        # Day 5: Freshness checks based on event timestamp and ttl_seconds.
        freshness_flag, freshness_multiplier = self._freshness_assessment(
            event_timestamp=getattr(canonical_record, "timestamp"),
            ingestion_timestamp_utc=ingestion_timestamp_utc,
            ttl_seconds=ttl_seconds,
        )
        if freshness_flag:
            sidecar_quality_flags.append(freshness_flag)
        adjusted_confidence *= freshness_multiplier

        manipulation_risk = self._clamp01(
            self._coerce_float(payload_dict.get("manipulation_risk_score"), default=0.0) + scam_risk_increment
        )
        if "potential_spam" in sidecar_quality_flags:
            manipulation_risk = max(manipulation_risk, 0.7)
            adjusted_confidence *= 0.5

        sidecar = TextSidecarMetadata(
            source_type=getattr(canonical_record, "source_type"),
            source_id=str(getattr(canonical_record, "source_id")),
            ingestion_timestamp_utc=ingestion_timestamp_utc,
            source_route_detail=record.source_route_detail,
            quality_flags=sidecar_quality_flags,
            manipulation_risk_score=manipulation_risk,
            confidence=min(max(adjusted_confidence, 0.0), 1.0),
            ttl_seconds=ttl_seconds,
            compliance_status=ComplianceStatus.ALLOW,
            compliance_reason=None,
        )
        return canonical_record, sidecar

    def evaluate_compliance(
        self,
        record: RawTextRecord,
        payload: Mapping[str, Any],
    ) -> ComplianceDecision:
        source_entry = self._allowlist.get(record.source_name)
        if source_entry is None:
            return self._reject("source_not_allowlisted")

        if not bool(source_entry.get("allowed", False)):
            return self._reject("source_blocked")

        configured_source_type = str(source_entry.get("source_type", ""))
        if configured_source_type != record.source_type.value:
            return self._reject("source_type_not_allowed")

        allowed_routes = set(source_entry.get("allowed_routes", []))
        if record.source_route_detail.value not in allowed_routes:
            return self._reject("route_not_allowed")

<<<<<<< HEAD
        # Day 3: Source-specific compliance checks.
=======
        if record.source_route_detail.value == "fallback_scraper":
            if not bool(source_entry.get("allow_fallback_scraper", False)):
                return self._reject("fallback_scraper_disabled")
            fallback_flag_field = str(source_entry.get("fallback_emergency_flag_field", "fallback_emergency_active"))
            if bool(source_entry.get("fallback_emergency_only", False)) and not bool(payload.get(fallback_flag_field, False)):
                return self._reject("fallback_requires_emergency")

        # Day 3: Source-specific compliance checks
>>>>>>> 701ccfb8293a2001f6b46632488e94f99447ad31
        compliance_checks = source_entry.get("compliance_checks", [])
        content_lower = record.content.lower()

        if any(neg in content_lower for neg in self._negative_filters):
            return self._reject("negative_filter_match")

        if "india_relevance" in compliance_checks:
            if not any(kw in content_lower for kw in self._india_relevance_keywords):
                return self._reject("low_india_relevance")

        if self._global_rules.get("reject_if_unpublished", False) and not bool(payload.get("is_published", True)):
            return self._reject("unpublished_content")

        if self._global_rules.get("reject_if_embargoed", False) and bool(payload.get("is_embargoed", False)):
            return self._reject("embargoed_content")

        if self._global_rules.get("reject_if_unlicensed", False) and not bool(payload.get("license_ok", True)):
            return self._reject("unlicensed_content")

        if self._global_rules.get("require_public_release_timestamp", False) and "timestamp" not in payload:
            return self._reject("missing_public_release_timestamp")

        if self._global_rules.get("require_source_url", False) and not payload.get("url"):
            return self._reject("missing_source_url")

        return ComplianceDecision(status=ComplianceStatus.ALLOW, reason=None)

    def _calculate_adjusted_confidence(
        self,
        record: RawTextRecord,
        payload: Mapping[str, Any],
        base_confidence: float,
    ) -> float:
        confidence = base_confidence

        # RSS feeds and official APIs get a boost.
        if record.source_type.value in ("rss_feed", "official_api"):
            confidence += 0.2

        # Fallback scrapers get a penalty.
        if record.source_route_detail.value == "fallback_scraper":
            confidence -= 0.3

        # Social media starts lower.
        if record.source_type.value == "social_media":
            confidence -= 0.1

        # X-specific engagement boosting.
        if record.source_name == "x_posts":
            likes = int(payload.get("likes", 0))
            if likes > 1000:
                confidence += 0.1
            elif likes < 10:
                confidence -= 0.1

        return confidence

    @staticmethod
    def _extract_quality_flags(payload: Mapping[str, Any], canonical_record: BaseModel) -> list[str]:
        quality_flags = TextualValidator._extract_payload_quality_flags(payload)

        quality_status = getattr(canonical_record, "quality_status", None)
        if quality_status is not None:
            quality_flags.append(str(getattr(quality_status, "value", quality_status)))

        return quality_flags

    @classmethod
    def _canonical_payload_only(cls, record_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = set(cls._BASE_CANONICAL_FIELDS)
        allowed_fields.update(cls._RECORD_TYPE_CANONICAL_FIELDS.get(record_type, set()))
<<<<<<< HEAD
        return {key: value for key, value in payload.items() if key in allowed_fields}
=======
        operational_keys = {
            "is_published",
            "is_embargoed",
            "license_ok",
            "manipulation_risk_score",
            "confidence",
            "quality_flags",
            "extraction_quality_score",
            "pdf_quality_status",
            "pdf_extracted_char_count",
            "normalized_content",
            "transliterated_content",
            "fallback_emergency_active",
        }
        return {
            key: value
            for key, value in payload.items()
            if key in allowed_fields and key not in operational_keys
        }
>>>>>>> 701ccfb8293a2001f6b46632488e94f99447ad31

    @staticmethod
    def _reject(reason: str) -> ComplianceDecision:
        return ComplianceDecision(status=ComplianceStatus.REJECT, reason=reason)

    @staticmethod
    def _extract_payload_quality_flags(payload: Mapping[str, Any]) -> list[str]:
        quality_flags: list[str] = []
        payload_flags = payload.get("quality_flags")
        if isinstance(payload_flags, list):
            for item in payload_flags:
                if isinstance(item, str):
                    quality_flags.append(item)
        return quality_flags

    def _extract_pdf_quality_score(self, payload: Mapping[str, Any]) -> float | None:
        raw_score = payload.get("extraction_quality_score")
        if raw_score is None:
            return None
        try:
            return self._clamp01(float(raw_score))
        except (TypeError, ValueError):
            return None

    def _pdf_quality_flags_for_payload(self, payload: Mapping[str, Any]) -> list[str]:
        extraction_score = self._extract_pdf_quality_score(payload)
        if extraction_score is None:
            return []
        if extraction_score < self._pdf_fail_below:
            return ["pdf_extraction_low_quality"]
        if extraction_score < self._pdf_warn_below:
            return ["pdf_extraction_warn"]
        return ["pdf_extraction_pass"]

    def _build_reject_sidecar(
        self,
        *,
        record: RawTextRecord,
        payload: Mapping[str, Any],
        reason: str,
        quality_flags: list[str],
        risk_increment: float,
    ) -> TextSidecarMetadata:
        reject_risk = self._clamp01(self._coerce_float(payload.get("manipulation_risk_score"), default=0.0) + risk_increment)
        if "potential_spam" in quality_flags:
            reject_risk = max(reject_risk, 0.7)

        return TextSidecarMetadata(
            source_type=record.source_type,
            source_id=record.source_id,
            ingestion_timestamp_utc=self._resolve_ingestion_timestamp(payload),
            source_route_detail=record.source_route_detail,
            quality_flags=quality_flags,
            manipulation_risk_score=reject_risk,
            confidence=0.0,
            ttl_seconds=0,
            compliance_status=ComplianceStatus.REJECT,
            compliance_reason=reason,
        )

    def _freshness_assessment(
        self,
        *,
        event_timestamp: datetime,
        ingestion_timestamp_utc: datetime,
        ttl_seconds: int,
    ) -> tuple[str, float]:
        if ttl_seconds <= 0 or event_timestamp.tzinfo is None:
            return "freshness_unknown", 1.0

        age_seconds = max(0.0, (ingestion_timestamp_utc.astimezone(UTC) - event_timestamp.astimezone(UTC)).total_seconds())
        if age_seconds <= ttl_seconds:
            return "freshness_fresh", 1.0
        if age_seconds <= ttl_seconds * 2:
            return "freshness_stale", 0.7
        return "freshness_expired", 0.4

    @classmethod
    def _required_fields_for_record_type(cls, record_type: str) -> set[str]:
        if record_type == "news_article":
            return {"headline", "publisher"}
        if record_type == "social_post":
            return {"platform"}
        if record_type == "earnings_transcript":
            return {"symbol", "quarter", "year"}
        return set()

    @classmethod
    def _coerce_quality_status(cls, raw_value: Any) -> str:
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in cls._QUALITY_STATUS_RANK:
                return normalized
        return "pass"

    @classmethod
    def _upgrade_quality_status(cls, current: str, candidate: str) -> str:
        if cls._QUALITY_STATUS_RANK.get(candidate, 0) > cls._QUALITY_STATUS_RANK.get(current, 0):
            return candidate
        return current

    @staticmethod
    def _resolve_ingestion_timestamp(payload: Mapping[str, Any]) -> datetime:
        raw_timestamp = payload.get("ingestion_timestamp_utc")
        if isinstance(raw_timestamp, datetime):
            if raw_timestamp.tzinfo is not None:
                return raw_timestamp.astimezone(UTC)
            return raw_timestamp.replace(tzinfo=UTC)
        if isinstance(raw_timestamp, str):
            try:
                parsed = datetime.fromisoformat(raw_timestamp)
                if parsed.tzinfo is not None:
                    return parsed.astimezone(UTC)
                return parsed.replace(tzinfo=UTC)
            except ValueError:
                pass
        return datetime.now(UTC)

    @staticmethod
    def _fingerprint_content(content: str) -> str:
        normalized = " ".join(content.lower().split())
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _has_duplicate_flag(flags: list[str]) -> bool:
        return "duplicate_source_id" in flags or "duplicate_content" in flags

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)
