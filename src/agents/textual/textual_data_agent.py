from __future__ import annotations
import logging
import json

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from src.agents.textual.adapters import (
    EconomicTimesAdapter,
    EarningsTranscriptAdapter,
    NSENewsAdapter,
    RBIReportsAdapter,
    RawTextRecord,
    TextSourceAdapter,
    XPostAdapter,
)
from src.agents.textual.cleaners import TextCleaner
from src.agents.textual.exporters import TextualExportBatch, TextualExporter
from src.agents.textual.validators import TextualValidator
from src.db.silver_db_recorder import SilverDBRecorder

DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "textual_data_agent_runtime_v1.json"
COMPLIANCE_LOG_PATH = Path(__file__).resolve().parents[3] / "logs" / "compliance_rejects.log"
PDF_SPOT_CHECK_REPORT_PATH = Path(__file__).resolve().parents[3] / "logs" / "textual_pdf_spot_check_report.json"

logger = logging.getLogger(__name__)


class TextualDataAgent:
    def __init__(
        self,
        adapters: Sequence[TextSourceAdapter],
        cleaner: TextCleaner,
        validator: TextualValidator,
        exporter: TextualExporter,
        recorder: SilverDBRecorder | None = None,
    ):
        self.adapters = list(adapters)
        self.cleaner = cleaner
        self.validator = validator
        self.exporter = exporter
        self.recorder = recorder
        self.last_pdf_spot_check_report: dict[str, Any] | None = None
        
        # Ensure log directory exists
        COMPLIANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        PDF_SPOT_CHECK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_default_components(
        cls,
        runtime_config_path: Path | None = None,
        recorder: SilverDBRecorder | None = None,
    ) -> "TextualDataAgent":
        config_path = runtime_config_path or DEFAULT_RUNTIME_CONFIG_PATH
        return cls(
            adapters=[
                NSENewsAdapter(),
                EconomicTimesAdapter(),
                RBIReportsAdapter(),
                EarningsTranscriptAdapter(),
                XPostAdapter(),
            ],
            cleaner=TextCleaner(),
            validator=TextualValidator.from_config_path(config_path),
            exporter=TextualExporter(),
            recorder=recorder,
        )

    def run_once(self, *, as_of_utc: datetime | None = None) -> TextualExportBatch:
        run_timestamp = as_of_utc or datetime.now(UTC)
        canonical_records: list[BaseModel] = []
        sidecar_records = []
        pdf_spot_checks: list[dict[str, Any]] = []

        for adapter in self.adapters:
            for raw_record in adapter.fetch(as_of_utc=run_timestamp):
                cleaned_record = self.cleaner.clean(raw_record)
                pdf_spot_check = self._build_pdf_spot_check(cleaned_record)
                if pdf_spot_check is not None:
                    pdf_spot_checks.append(pdf_spot_check)
                canonical_payload = self._build_canonical_payload(cleaned_record)
                canonical_record, sidecar_record = self.validator.validate_record(
                    cleaned_record,
                    canonical_payload,
                )
                
                if canonical_record is not None:
                    canonical_records.append(canonical_record)
                else:
                    self._log_compliance_rejection(sidecar_record)
                sidecar_records.append(sidecar_record)

        # Day 3: Batch-level post-processing (Burst Detection)
        self._detect_social_bursts(sidecar_records)
        self._persist_pdf_spot_check_report(run_timestamp, pdf_spot_checks)

        if self.recorder and canonical_records:
            from src.schemas.text_data import TextDataBase
            # Cast for type checker if needed, SilverDBRecorder expects List[TextDataBase]
            self.recorder.save_text_items(canonical_records)  # type: ignore

        return self.exporter.build_batch(canonical_records, sidecar_records)

    def _detect_social_bursts(self, sidecars: list[Any]) -> None:
        """Simple burst detection: if many posts share hashtags/keywords, increase risk."""
        social_indices = [
            i for i, s in enumerate(sidecars) 
            if s.source_type.value == "social_media" and s.compliance_status.value == "allow"
        ]
        
        if len(social_indices) < 5:
            return
            
        for i in social_indices:
            s = sidecars[i]
            updated_flags = list(s.quality_flags)
            if "high_volume_burst" not in updated_flags:
                updated_flags.append("high_volume_burst")
                
            sidecars[i] = s.model_copy(update={
                "quality_flags": updated_flags,
                "manipulation_risk_score": min(1.0, s.manipulation_risk_score + 0.2)
            })

    def _log_compliance_rejection(self, sidecar: Any) -> None:
        """Logs compliance rejection details to a dedicated file."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "source_id": sidecar.source_id,
            "source_type": sidecar.source_type,
            "reason": sidecar.compliance_reason,
            "route": sidecar.source_route_detail,
        }
        with open(COMPLIANCE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{log_entry}\n")

    def _persist_pdf_spot_check_report(
        self,
        run_timestamp: datetime,
        spot_checks: list[dict[str, Any]],
    ) -> None:
        if not spot_checks:
            self.last_pdf_spot_check_report = None
            return

        warn_below = float(self.validator.runtime_config.get("pdf_quality", {}).get("warn_below", 0.8))
        fail_below = float(self.validator.runtime_config.get("pdf_quality", {}).get("fail_below", 0.6))
        quality_scores = [float(item["quality_score"]) for item in spot_checks]
        report = {
            "generated_at_utc": run_timestamp.astimezone(UTC).isoformat(),
            "total_documents": len(spot_checks),
            "average_quality_score": round(sum(quality_scores) / len(quality_scores), 3),
            "minimum_quality_score": round(min(quality_scores), 3),
            "maximum_quality_score": round(max(quality_scores), 3),
            "warn_below": warn_below,
            "fail_below": fail_below,
            "warn_count": sum(1 for score in quality_scores if fail_below <= score < warn_below),
            "fail_count": sum(1 for score in quality_scores if score < fail_below),
            "spot_checks": spot_checks,
        }
        self.last_pdf_spot_check_report = report
        with open(PDF_SPOT_CHECK_REPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    @staticmethod
    def _build_pdf_spot_check(record: RawTextRecord) -> dict[str, Any] | None:
        raw_score = record.payload.get("extraction_quality_score")
        if raw_score is None:
            return None
        try:
            quality_score = min(max(float(raw_score), 0.0), 1.0)
        except (TypeError, ValueError):
            return None
        quality_status = "pass"
        if quality_score < 0.6:
            quality_status = "fail"
        elif quality_score < 0.8:
            quality_status = "warn"

        return {
            "source_id": record.source_id,
            "source_name": record.source_name,
            "record_type": record.record_type,
            "quality_score": quality_score,
            "quality_status": quality_status,
            "content_length_chars": len(record.content),
        }

    @staticmethod
    def _build_canonical_payload(record: RawTextRecord) -> dict[str, Any]:
        payload = dict(record.payload)
        canonical_payload: dict[str, Any] = {
            "source_id": record.source_id,
            "timestamp": record.timestamp,
            "content": record.content,
            "source_type": record.source_type.value,
        }

        for optional_field in (
            "url",
            "author",
            "language",
            "ingestion_timestamp_utc",
            "ingestion_timestamp_ist",
            "schema_version",
            "quality_status",
        ):
            if optional_field in payload:
                canonical_payload[optional_field] = payload[optional_field]

        if record.record_type == "news_article":
            canonical_payload["headline"] = str(payload.get("headline", ""))
            canonical_payload["publisher"] = str(payload.get("publisher", ""))
        elif record.record_type == "social_post":
            canonical_payload["platform"] = str(payload.get("platform", "X"))
            canonical_payload["likes"] = int(payload.get("likes", 0))
            canonical_payload["shares"] = int(payload.get("shares", 0))
        elif record.record_type == "earnings_transcript":
            canonical_payload["symbol"] = str(payload.get("symbol", ""))
            canonical_payload["quarter"] = str(payload.get("quarter", ""))
            canonical_payload["year"] = int(payload.get("year", datetime.now(UTC).year))
        else:
            raise ValueError(f"Unsupported record_type={record.record_type}")

        # Keep compliance-only fields for validator checks but never pass them to canonical validation.
        for operational_key in (
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
        ):
            if operational_key in payload:
                canonical_payload[operational_key] = payload[operational_key]

        return canonical_payload
