from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from src.agents.textual.adapters import RawTextRecord
from src.agents.textual.services.language_service import LanguageService

_WHITESPACE_RE = re.compile(r"\s+")


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

        cleaned_payload["quality_flags"] = self._dedupe_flags(quality_flags)

        return replace(record, content=cleaned_content, payload=cleaned_payload)

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
