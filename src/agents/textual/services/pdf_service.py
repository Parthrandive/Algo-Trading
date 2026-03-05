from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PDFExtractionResult:
    text: str
    quality_score: float
    quality_status: str
    metrics: dict[str, Any]


class PDFExtractor:
    """
    Extracts text and quality diagnostics from PDF bytes.
    This remains lightweight/mock-compatible while exposing production-like metrics.
    """

    WARN_THRESHOLD = 0.8
    FAIL_THRESHOLD = 0.6

    def extract(self, pdf_bytes: bytes) -> PDFExtractionResult:
        text = self.extract_text(pdf_bytes)
        quality_score = self.get_extraction_quality_score(pdf_bytes, extracted_text=text)
        quality_status = self._quality_status(quality_score)
        metrics = {
            "pdf_bytes": len(pdf_bytes),
            "extracted_char_count": len(text),
            "quality_score": quality_score,
            "quality_status": quality_status,
        }
        return PDFExtractionResult(
            text=text,
            quality_score=quality_score,
            quality_status=quality_status,
            metrics=metrics,
        )

    def extract_text(self, pdf_bytes: bytes) -> str:
        logger.info("PDFExtractor: extracting text from %s bytes", len(pdf_bytes))
        if b"RBI" in pdf_bytes:
            return (
                "RBI Bulletin: Macroeconomic indicators show stable growth. "
                "Repo rate remains unchanged and liquidity conditions remain orderly."
            )
        if b"INFY" in pdf_bytes:
            return (
                "Infosys Earnings Transcript: Q3 revenue up 5%. "
                "Strong growth in digital services and resilient margin commentary."
            )
        return "Generic extracted text from PDF document."

    def get_extraction_quality_score(self, pdf_bytes: bytes, *, extracted_text: str | None = None) -> float:
        """Returns a bounded extraction quality score (0.0 to 1.0)."""
        text = extracted_text if extracted_text is not None else self.extract_text(pdf_bytes)

        score = 0.5
        if pdf_bytes.startswith(b"%PDF"):
            score += 0.1
        if len(text) >= 80:
            score += 0.2
        if "Generic extracted text" not in text:
            score += 0.15
        if any(keyword in text for keyword in ("RBI", "Earnings Transcript", "revenue", "Repo rate")):
            score += 0.05

        bounded = max(0.0, min(score, 1.0))
        return round(bounded, 3)

    def spot_check(self, source_id: str, pdf_bytes: bytes) -> dict[str, Any]:
        result = self.extract(pdf_bytes)
        return {
            "source_id": source_id,
            "quality_score": result.quality_score,
            "quality_status": result.quality_status,
            "pdf_bytes": result.metrics["pdf_bytes"],
            "extracted_char_count": result.metrics["extracted_char_count"],
            "passes_threshold": result.quality_score >= self.FAIL_THRESHOLD,
        }

    def _quality_status(self, quality_score: float) -> str:
        if quality_score < self.FAIL_THRESHOLD:
            return "fail"
        if quality_score < self.WARN_THRESHOLD:
            return "warn"
        return "pass"
