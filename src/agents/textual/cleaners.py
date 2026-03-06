from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from src.agents.textual.adapters import RawTextRecord
from src.agents.textual.services.language_service import LanguageService

_WHITESPACE_RE = re.compile(r"\s+")

# Basic Hinglish -> English Transliterator (Stub for Phase 1)
_HINGLISH_MAP = {
    r"\bbhai\b": "brother",
    r"\bbull run hai\b": "it is a bull run",
    r"\bgira\b": "fell",
    r"\buga\b": "rose",
    r"\bprofit book karlo\b": "book your profits",
    r"\bhold karo\b": "hold it",
    r"\bbakwas\b": "nonsense",
    r"\bfake hai\b": "is fake",
}

# Slang/Scam lexicon for manipulation detection
_SLANG_SCAM_LEXICON = [
    r"\b100x\b",
    r"\bguaranteed returns\b",
    r"\bmultibagger\b",
    r"\bpump and dump\b",
    r"\btns\b",
    r"\bsure shot\b",
    r"\bjackpot\b",
]

class TextCleaner:
    def __init__(self, language_service: LanguageService | None = None):
        self.language_service = language_service or LanguageService()

    def clean(self, record: RawTextRecord) -> RawTextRecord:
        cleaned_content = self.normalize_text(record.content)
        language = self.language_service.detect_language(cleaned_content)

        cleaned_payload = dict(record.payload)
        cleaned_payload["content"] = cleaned_content
        cleaned_payload["language"] = language

        quality_flags = self._coerce_quality_flags(cleaned_payload.get("quality_flags"))

        if language == "code_mixed":
            cleaned_payload["normalized_content"] = self.language_service.normalize_hinglish(cleaned_content)
            quality_flags.append("code_mixed_detected")
        elif language == "hi":
            quality_flags.append("hindi_detected")

        if language in {"hi", "code_mixed"}:
            transliterated = self.language_service.transliterate_to_latin(cleaned_content)
            cleaned_payload["transliterated_content"] = transliterated
            if transliterated != cleaned_content:
                quality_flags.append("transliteration_applied")

        # Add Slang/Scam Hooks (from feature branch)
        scam_score = self._compute_scam_score(cleaned_content)
        if scam_score > 0:
            quality_flags.append("scam_slang_detected")
            current_risk = float(cleaned_payload.get("manipulation_risk_score", 0.0))
            cleaned_payload["manipulation_risk_score"] = min(1.0, current_risk + (scam_score * 0.2))

        cleaned_payload["quality_flags"] = self._dedupe_flags(quality_flags)

        return replace(record, content=cleaned_content, payload=cleaned_payload)

    @staticmethod
    def _compute_scam_score(text: str) -> float:
        lower_text = text.lower()
        hits = sum(1 for pattern in _SLANG_SCAM_LEXICON if re.search(pattern, lower_text))
        return min(1.0, hits / 5.0)

    @staticmethod
    def normalize_text(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _coerce_quality_flags(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _dedupe_flags(flags: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for flag in flags:
            normalized = flag.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped
