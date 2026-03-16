from __future__ import annotations

import re

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


class SentimentLanguageService:
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
        if not text.strip():
            return "en"

        has_devanagari = bool(_DEVANAGARI_RE.search(text))
        has_latin = bool(_LATIN_RE.search(text))
        if has_devanagari and has_latin:
            return "code_mixed"
        if has_devanagari:
            return "hi"

        tokens = {token.lower() for token in _TOKEN_RE.findall(text.lower())}
        has_roman_hindi = bool(tokens & self.HINGLISH_KEYWORDS)
        has_market_terms = bool(tokens & self.ENGLISH_MARKET_TERMS)
        if has_roman_hindi and has_market_terms:
            return "code_mixed"
        if has_roman_hindi:
            return "hi"
        return "en"

    def normalize_hinglish(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        for slang, normalized_token in self._SLANG_NORMALIZATION.items():
            normalized = re.sub(rf"\b{re.escape(slang)}\b", normalized_token, normalized)
        return normalized

    def transliterate_to_latin(self, text: str) -> str:
        transliterated = "".join(self._DEVANAGARI_TO_LATIN.get(ch, ch) for ch in text)
        return re.sub(r"\s+", " ", transliterated).strip()


class SentimentSafetyService:
    SCAM_LEXICON = frozenset(
        {
            "guaranteed returns",
            "jackpot tips",
            "double your investment",
            "pump and dump",
            "insider info",
            "100% success rate",
            "no risk",
            "get rich quick",
            "signal group",
            "operator call",
            "tip subscription",
            "guaranteed profit",
        }
    )
    HINGLISH_SCAM_LEXICON = frozenset(
        {
            "paisa double",
            "loss nahi",
            "pakka profit",
            "tip le lo",
            "operator se call",
        }
    )

    _EXCLAMATION_SPAM_RE = re.compile(r"[!]{3,}")
    _PUNCTUATION_BURST_RE = re.compile(r"[!?]{5,}")
    _URL_RE = re.compile(r"https?://", re.IGNORECASE)
    _PROMOTIONAL_CLAIM_RE = re.compile(
        r"\b(?:\d{2,4}%\s*returns?|guaranteed|risk[- ]?free|no risk|double your investment)\b",
        re.IGNORECASE,
    )

    def check_for_scams(
        self,
        text: str,
        *,
        normalized_text: str | None = None,
        transliterated_text: str | None = None,
    ) -> tuple[list[str], float]:
        corpus_parts = [text.lower()]
        if normalized_text:
            corpus_parts.append(normalized_text.lower())
        if transliterated_text:
            corpus_parts.append(transliterated_text.lower())
        corpus = " ".join(corpus_parts)

        matched_patterns: list[str] = []
        for pattern in sorted(self.SCAM_LEXICON | self.HINGLISH_SCAM_LEXICON):
            if pattern in corpus:
                matched_patterns.append(pattern)

        flags = [f"scam_pattern:{pattern.replace(' ', '_')}" for pattern in matched_patterns]
        risk_increase = min(1.0, 0.2 * len(matched_patterns))
        if len(matched_patterns) >= 3:
            risk_increase = min(1.0, risk_increase + 0.1)
        return flags, risk_increase

    def check_for_adversarial_patterns(self, text: str) -> tuple[list[str], float]:
        stripped = text.strip()
        if not stripped:
            return ["missing_content"], 0.0

        lowered = stripped.lower()
        flags: list[str] = []
        risk_increase = 0.0

        if self._PUNCTUATION_BURST_RE.search(stripped):
            flags.append("adversarial_punctuation_burst")
            risk_increase += 0.1
        elif self._EXCLAMATION_SPAM_RE.search(stripped):
            flags.append("adversarial_exclamation_spam")
            risk_increase += 0.05

        url_count = len(self._URL_RE.findall(stripped))
        if url_count >= 3:
            flags.append("link_spam")
            risk_increase += 0.1

        if self._PROMOTIONAL_CLAIM_RE.search(lowered):
            flags.append("promotional_claim_pattern")
            risk_increase += 0.15

        alpha_chars = [ch for ch in stripped if ch.isalpha()]
        if len(alpha_chars) >= 20:
            uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
            if uppercase_ratio >= 0.65:
                flags.append("adversarial_all_caps")
                risk_increase += 0.1

        return flags, min(1.0, risk_increase)
