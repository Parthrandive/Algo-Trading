from __future__ import annotations

import re
from dataclasses import replace

from src.agents.textual.adapters import RawTextRecord
from src.agents.textual.services.language_service import LanguageService

_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    def __init__(self):
        self.language_service = LanguageService()

    def clean(self, record: RawTextRecord) -> RawTextRecord:
        cleaned_content = self.normalize_text(record.content)
        language = self.language_service.detect_language(cleaned_content)
        
        cleaned_payload = dict(record.payload)
        cleaned_payload["content"] = cleaned_content
        cleaned_payload["language"] = language
        
        # If Hinglish, maybe add normalized version to payload for downstream ingestion
        if language == "code_mixed":
            cleaned_payload["normalized_content"] = self.language_service.normalize_hinglish(cleaned_content)
            
        return replace(record, content=cleaned_content, payload=cleaned_payload)

    @staticmethod
    def normalize_text(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text).strip()
