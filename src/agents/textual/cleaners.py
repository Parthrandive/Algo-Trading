from __future__ import annotations

import re
from dataclasses import replace

from src.agents.textual.adapters import RawTextRecord

_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    def clean(self, record: RawTextRecord) -> RawTextRecord:
        cleaned_content = self.normalize_text(record.content)
        cleaned_payload = dict(record.payload)
        cleaned_payload["content"] = cleaned_content
        return replace(record, content=cleaned_content, payload=cleaned_payload)

    @staticmethod
    def normalize_text(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text).strip()
