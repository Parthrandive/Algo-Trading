from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from src.agents.textual.adapters import RawTextRecord
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
    def __init__(self, runtime_config: Mapping[str, Any]):
        self.runtime_config = dict(runtime_config)
        self._allowlist = {
            str(entry["source_name"]): dict(entry)
            for entry in self.runtime_config.get("source_allowlist", [])
        }
        self._canonical_schema_keys = dict(self.runtime_config.get("canonical_schema_keys", {}))
        self._default_ttl_seconds = dict(self.runtime_config.get("default_ttl_seconds", {}))
        self._global_rules = dict(self.runtime_config.get("global_compliance_rules", {}))

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
        decision = self.evaluate_compliance(record, payload_dict)

        if not decision.allowed:
            sidecar = TextSidecarMetadata(
                source_type=record.source_type,
                source_id=record.source_id,
                ingestion_timestamp_utc=datetime.now(UTC),
                source_route_detail=record.source_route_detail,
                quality_flags=["compliance_reject"],
                manipulation_risk_score=0.0,
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
        ingestion_timestamp_utc = getattr(canonical_record, "ingestion_timestamp_utc")

        sidecar = TextSidecarMetadata(
            source_type=getattr(canonical_record, "source_type"),
            source_id=str(getattr(canonical_record, "source_id")),
            ingestion_timestamp_utc=ingestion_timestamp_utc,
            source_route_detail=record.source_route_detail,
            quality_flags=sidecar_quality_flags,
            manipulation_risk_score=float(payload_dict.get("manipulation_risk_score", 0.0)),
            confidence=float(payload_dict.get("confidence", 0.5)),
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

    @staticmethod
    def _extract_quality_flags(payload: Mapping[str, Any], canonical_record: BaseModel) -> list[str]:
        quality_flags: list[str] = []

        payload_flags = payload.get("quality_flags")
        if isinstance(payload_flags, list):
            for item in payload_flags:
                if isinstance(item, str):
                    quality_flags.append(item)

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
        }
        return {key: value for key, value in payload.items() if key not in operational_keys}

    @staticmethod
    def _reject(reason: str) -> ComplianceDecision:
        return ComplianceDecision(status=ComplianceStatus.REJECT, reason=reason)
