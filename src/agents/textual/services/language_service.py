from __future__ import annotations

import re

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


class LanguageService:
    """Detects and normalizes English/Hindi/code-mixed textual inputs."""

    HINGLISH_KEYWORDS = frozenset(
        {
            "bhai",
            "paisa",
            "kya",
            "lagta",
            "hoga",
            "girega",
            "jayega",
            "lelo",
            "karo",
            "aaj",
            "kal",
        }
    )
    ENGLISH_MARKET_TERMS = frozenset(
        {
            "market",
            "nifty",
            "sensex",
            "stock",
            "rally",
            "selloff",
            "profit",
            "loss",
            "trade",
            "buy",
            "sell",
        }
    )
    _SLANG_NORMALIZATION = {
        "paisa": "money",
        "girega": "will fall",
        "gira": "fell",
        "jayega": "will move",
        "kya": "what",
        "lagta": "seems",
        "hoga": "will be",
        "lelo": "take",
        "karo": "do",
        "bhai": "brother",
        "profit book karlo": "book your profits",
        "fake hai": "is fake",
    }
    _DEVANAGARI_TO_LATIN = {
        "अ": "a",
        "आ": "aa",
        "इ": "i",
        "ई": "ii",
        "उ": "u",
        "ऊ": "uu",
        "ए": "e",
        "ऐ": "ai",
        "ओ": "o",
        "औ": "au",
        "क": "k",
        "ख": "kh",
        "ग": "g",
        "घ": "gh",
        "च": "ch",
        "छ": "chh",
        "ज": "j",
        "झ": "jh",
        "ट": "t",
        "ठ": "th",
        "ड": "d",
        "ढ": "dh",
        "ण": "n",
        "त": "t",
        "थ": "th",
        "द": "d",
        "ध": "dh",
        "न": "n",
        "प": "p",
        "फ": "ph",
        "ब": "b",
        "भ": "bh",
        "म": "m",
        "य": "y",
        "र": "r",
        "ल": "l",
        "व": "v",
        "श": "sh",
        "ष": "sh",
        "स": "s",
        "ह": "h",
        "ा": "a",
        "ि": "i",
        "ी": "ii",
        "ु": "u",
        "ू": "uu",
        "े": "e",
        "ै": "ai",
        "ो": "o",
        "ौ": "au",
        "ं": "n",
        "ँ": "n",
        "ः": "h",
        "्": "",
    }

    def detect_language(self, text: str) -> str:
        """Returns one of: `en`, `hi`, `code_mixed`."""
        if not text.strip():
            return "en"

        text_lower = text.lower()
        has_devanagari = bool(_DEVANAGARI_RE.search(text))
        has_latin = bool(_LATIN_RE.search(text))

        if has_devanagari and has_latin:
            return "code_mixed"
        if has_devanagari:
            return "hi"

        tokens = {token.lower() for token in _TOKEN_RE.findall(text_lower)}
        has_roman_hindi = bool(tokens & self.HINGLISH_KEYWORDS)
        has_english_market_terms = bool(tokens & self.ENGLISH_MARKET_TERMS)

        if has_roman_hindi and has_english_market_terms:
            return "code_mixed"
        if has_roman_hindi:
            return "hi"
        return "en"

    def normalize_hinglish(self, text: str) -> str:
        """Normalizes common market Hinglish slang to stable tokens."""
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        for slang, normalized_token in self._SLANG_NORMALIZATION.items():
            normalized = re.sub(rf"\b{re.escape(slang)}\b", normalized_token, normalized)
        return normalized

    def transliterate_to_latin(self, text: str) -> str:
        """Best-effort transliteration for Devanagari characters."""
        transliterated = "".join(self._DEVANAGARI_TO_LATIN.get(ch, ch) for ch in text)
        return re.sub(r"\s+", " ", transliterated).strip()
