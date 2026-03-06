from __future__ import annotations

import re
from dataclasses import replace

from src.agents.textual.adapters import RawTextRecord

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
    def clean(self, record: RawTextRecord) -> RawTextRecord:
        content = record.content
        payload = dict(record.payload)
        
        # 1. Detect code-mixing (Hinglish)
        if self._is_hinglish(content):
            payload["language"] = "code_mixed"
            content = self._transliterate_hinglish(content)
            
        # 2. Add Slang/Scam Hooks
        scam_score = self._compute_scam_score(content)
        if scam_score > 0:
            payload["quality_flags"] = payload.get("quality_flags", []) + ["scam_slang_detected"]
            # Escalate manipulation risk slightly
            current_risk = float(payload.get("manipulation_risk_score", 0.0))
            payload["manipulation_risk_score"] = min(1.0, current_risk + (scam_score * 0.2))

        cleaned_content = self.normalize_text(content)
        payload["content"] = cleaned_content
        return replace(record, content=cleaned_content, payload=payload)

    @staticmethod
    def _is_hinglish(text: str) -> bool:
        lower_text = text.lower()
        return any(re.search(pattern, lower_text) for pattern in _HINGLISH_MAP.keys())

    @staticmethod
    def _transliterate_hinglish(text: str) -> str:
        # A lightweight regex replacement for Phase 1
        res = text.lower()
        for pattern, replacement in _HINGLISH_MAP.items():
            res = re.sub(pattern, replacement, res)
        return res

    @staticmethod
    def _compute_scam_score(text: str) -> float:
        lower_text = text.lower()
        hits = sum(1 for pattern in _SLANG_SCAM_LEXICON if re.search(pattern, lower_text))
        # Cap score at 1.0 (5 hits = 1.0)
        return min(1.0, hits / 5.0)

    @staticmethod
    def normalize_text(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text).strip()
