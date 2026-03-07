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
        "\u0905": "a",
        "\u0906": "aa",
        "\u0907": "i",
        "\u0908": "ii",
        "\u0909": "u",
        "\u090a": "uu",
        "\u090f": "e",
        "\u0910": "ai",
        "\u0913": "o",
        "\u0914": "au",
        "\u0915": "k",
        "\u0916": "kh",
        "\u0917": "g",
        "\u0918": "gh",
        "\u091a": "ch",
        "\u091b": "chh",
        "\u091c": "j",
        "\u091d": "jh",
        "\u091f": "t",
        "\u0920": "th",
        "\u0921": "d",
        "\u0922": "dh",
        "\u0923": "n",
        "\u0924": "t",
        "\u0925": "th",
        "\u0926": "d",
        "\u0927": "dh",
        "\u0928": "n",
        "\u092a": "p",
        "\u092b": "ph",
        "\u092c": "b",
        "\u092d": "bh",
        "\u092e": "m",
        "\u092f": "y",
        "\u0930": "r",
        "\u0932": "l",
        "\u0935": "v",
        "\u0936": "sh",
        "\u0937": "sh",
        "\u0938": "s",
        "\u0939": "h",
        "\u093e": "a",
        "\u093f": "i",
        "\u0940": "ii",
        "\u0941": "u",
        "\u0942": "uu",
        "\u0947": "e",
        "\u0948": "ai",
        "\u094b": "o",
        "\u094c": "au",
        "\u0902": "n",
        "\u0901": "n",
        "\u0903": "h",
        "\u094d": "",
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
