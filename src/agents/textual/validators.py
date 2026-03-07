from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from src.agents.textual.adapters import RawTextRecord
from src.schemas.text_sidecar import ComplianceStatus, TextSidecarMetadata
from src.utils.schema_registry import SchemaRegistry
from src.agents.textual.services.safety_service import SafetyService


@dataclass(frozen=True)
class ComplianceDecision:
    status: ComplianceStatus
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.status == ComplianceStatus.ALLOW


class TextualValidator:
    def __init__(self, runtime_config: Mapping[str, Any]):
        self.runtime_config = dict(runtime_config)
        self._allowlist = {
            str(entry["source_name"]): dict(entry)
            for entry in self.runtime_config.get("source_allowlist", [])
        }
        self._canonical_schema_keys = dict(self.runtime_config.get("canonical_schema_keys", {}))
        self._default_ttl_seconds = dict(self.runtime_config.get("default_ttl_seconds", {}))
        self._global_rules = dict(self.runtime_config.get("global_compliance_rules", {}))
        
        # Day 3: X-specific templates and filters
        x_templates = self.runtime_config.get("x_query_templates", {})
        self._india_relevance_keywords = {"nifty", "sensex", "india", "rbi", "mpc", "repo", "nse", "infy", "tcs", "hdfc", "reliance"}
        self._negative_filters = set(x_templates.get("negative_filters", []))

        # Day 4: Safety and PDF scoring controls
        self.safety_service = SafetyService()
        pdf_quality_config = dict(self.runtime_config.get("pdf_quality", {}))
        self._pdf_warn_below = min(max(float(pdf_quality_config.get("warn_below", 0.8)), 0.0), 1.0)
        self._pdf_fail_below = min(max(float(pdf_quality_config.get("fail_below", 0.6)), 0.0), 1.0)
        if self._pdf_fail_below > self._pdf_warn_below:
            self._pdf_fail_below, self._pdf_warn_below = self._pdf_warn_below, self._pdf_fail_below

        quality_controls = dict(self.runtime_config.get("quality_controls", {}))
        self._min_content_chars = max(int(quality_controls.get("min_content_chars", 20)), 0)
        self._max_content_chars = max(int(quality_controls.get("max_content_chars", 30000)), self._min_content_chars + 1)
        self._max_future_skew_seconds = max(int(quality_controls.get("max_future_skew_seconds", 300)), 0)
        freshness_windows = quality_controls.get("freshness_windows_seconds", {})
        if isinstance(freshness_windows, dict):
            self._freshness_windows_seconds = {str(k): max(int(v), 0) for k, v in freshness_windows.items()}
        else:
            self._freshness_windows_seconds = {}

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

    def validate_record(
        self,
        record: RawTextRecord,
        payload: Mapping[str, Any],
    ) -> tuple[BaseModel | None, TextSidecarMetadata]:
        payload_dict = dict(payload)
        scam_flags, scam_risk_increment = self.safety_service.check_for_scams(
            record.content,
            normalized_text=payload_dict.get("normalized_content") if isinstance(payload_dict.get("normalized_content"), str) else None,
            transliterated_text=payload_dict.get("transliterated_content") if isinstance(payload_dict.get("transliterated_content"), str) else None,
        )
        pdf_quality_flags = self._pdf_quality_flags_for_payload(payload_dict)
        extraction_score = self._extract_pdf_quality_score(payload_dict)
        quality_gate_flags, quality_gate_reject_reason = self._evaluate_quality_gate(record, payload_dict)
        if quality_gate_reject_reason is not None:
            reject_quality_flags = ["quality_gate_reject"]
            reject_quality_flags.extend(self._extract_payload_quality_flags(payload_dict))
            reject_quality_flags.extend(quality_gate_flags)
            reject_quality_flags.extend(pdf_quality_flags)
            reject_quality_flags.extend(scam_flags)

            reject_risk = self._clamp01(float(payload_dict.get("manipulation_risk_score", 0.0)) + scam_risk_increment)
            if "potential_spam" in reject_quality_flags:
                reject_risk = max(reject_risk, 0.7)

            sidecar = TextSidecarMetadata(
                source_type=record.source_type,
                source_id=record.source_id,
                ingestion_timestamp_utc=datetime.now(UTC),
                source_route_detail=record.source_route_detail,
                quality_flags=reject_quality_flags,
                manipulation_risk_score=reject_risk,
                confidence=0.0,
                ttl_seconds=0,
                compliance_status=ComplianceStatus.REJECT,
                compliance_reason=quality_gate_reject_reason,
            )
            return None, sidecar

        decision = self.evaluate_compliance(record, payload_dict)

        if not decision.allowed:
            reject_quality_flags = ["compliance_reject"]
            reject_quality_flags.extend(self._extract_payload_quality_flags(payload_dict))
            reject_quality_flags.extend(quality_gate_flags)
            reject_quality_flags.extend(pdf_quality_flags)
            reject_quality_flags.extend(scam_flags)

            reject_risk = self._clamp01(float(payload_dict.get("manipulation_risk_score", 0.0)) + scam_risk_increment)
            if "potential_spam" in reject_quality_flags:
                reject_risk = max(reject_risk, 0.7)

            sidecar = TextSidecarMetadata(
                source_type=record.source_type,
                source_id=record.source_id,
                ingestion_timestamp_utc=datetime.now(UTC),
                source_route_detail=record.source_route_detail,
                quality_flags=reject_quality_flags,
                manipulation_risk_score=reject_risk,
                confidence=0.0,
                ttl_seconds=0,
                compliance_status=decision.status,
                compliance_reason=decision.reason,
            )
            return None, sidecar

        canonical_record = self.validate_canonical(
            record.record_type,
            self._canonical_payload_only(payload_dict),
        )
        sidecar_quality_flags = self._extract_quality_flags(payload_dict, canonical_record)
        sidecar_quality_flags.extend(quality_gate_flags)
        sidecar_quality_flags.extend(pdf_quality_flags)
        sidecar_quality_flags.extend(scam_flags)
        ingestion_timestamp_utc = getattr(canonical_record, "ingestion_timestamp_utc")

        # Day 3: Reliability and confidence adjustments
        base_confidence = float(payload_dict.get("confidence", 0.5))
        adjusted_confidence = self._calculate_adjusted_confidence(record, payload_dict, base_confidence)

        # Day 4: PDF extraction quality should downgrade confidence on weak extraction.
        if extraction_score is not None:
            adjusted_confidence *= extraction_score
            if extraction_score < self._pdf_fail_below:
                adjusted_confidence *= 0.8

        manipulation_risk = self._clamp01(float(payload_dict.get("manipulation_risk_score", 0.0)) + scam_risk_increment)
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
            ttl_seconds=self.default_ttl_seconds(record.record_type),
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

        if record.source_route_detail.value == "fallback_scraper":
            if not bool(source_entry.get("allow_fallback_scraper", False)):
                return self._reject("fallback_scraper_disabled")
            fallback_flag_field = str(source_entry.get("fallback_emergency_flag_field", "fallback_emergency_active"))
            if bool(source_entry.get("fallback_emergency_only", False)) and not bool(payload.get(fallback_flag_field, False)):
                return self._reject("fallback_requires_emergency")

        # Day 3: Source-specific compliance checks
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
        base_confidence: float
    ) -> float:
        confidence = base_confidence
        
        # RSS feeds and official APIs get a boost
        if record.source_type.value in ("rss_feed", "official_api"):
            confidence += 0.2
            
        # Fallback scrapers get a penalty
        if record.source_route_detail.value == "fallback_scraper":
            confidence -= 0.3
            
        # Social media starts lower
        if record.source_type.value == "social_media":
            confidence -= 0.1
            
        # X-specific engagement boosting
        if record.source_name == "x_posts":
            likes = int(payload.get("likes", 0))
            if likes > 1000:
                confidence += 0.1
            elif likes < 10:
                confidence -= 0.1
                
        return confidence

    def _evaluate_quality_gate(
        self,
        record: RawTextRecord,
        payload: Mapping[str, Any],
    ) -> tuple[list[str], str | None]:
        quality_flags: list[str] = []
        missing_fields: list[str] = []

        if not record.source_id.strip():
            missing_fields.append("source_id")
        if not record.content.strip():
            missing_fields.append("content")
        if record.timestamp is None:
            missing_fields.append("timestamp")
        if missing_fields:
            quality_flags.extend(f"missing_field:{field}" for field in missing_fields)
            return quality_flags, "missing_required_fields"

        if not isinstance(record.timestamp, datetime) or record.timestamp.tzinfo is None:
            quality_flags.append("malformed_timestamp")
            return quality_flags, "malformed_timestamp"

        timestamp_utc = record.timestamp.astimezone(UTC)
        now_utc = datetime.now(UTC)
        if timestamp_utc > now_utc + timedelta(seconds=self._max_future_skew_seconds):
            quality_flags.append("timestamp_in_future")
            return quality_flags, "malformed_timestamp"

        content_len = len(record.content.strip())
        if content_len < self._min_content_chars:
            quality_flags.append("content_too_short")
        if content_len > self._max_content_chars:
            quality_flags.append("content_length_outlier")

        freshness_window_seconds = self._freshness_windows_seconds.get(record.record_type)
        if freshness_window_seconds:
            if now_utc - timestamp_utc > timedelta(seconds=freshness_window_seconds):
                quality_flags.append("stale_timestamp")

        raw_quality_status = payload.get("quality_status")
        if isinstance(raw_quality_status, str) and raw_quality_status.strip().lower() == "fail":
            quality_flags.append("source_quality_fail")

        return quality_flags, None

    @staticmethod
    def _extract_quality_flags(payload: Mapping[str, Any], canonical_record: BaseModel) -> list[str]:
        quality_flags = TextualValidator._extract_payload_quality_flags(payload)

        quality_status = getattr(canonical_record, "quality_status", None)
        if quality_status is not None:
            quality_flags.append(str(getattr(quality_status, "value", quality_status)))

        return quality_flags

    @staticmethod
    def _canonical_payload_only(payload: Mapping[str, Any]) -> dict[str, Any]:
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
        return {key: value for key, value in payload.items() if key not in operational_keys}

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

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)
