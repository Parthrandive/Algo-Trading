from __future__ import annotations

import re


class SafetyService:
    """
    Service for detecting manipulation, spam, and slang-based scams.
    """
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
        """
        Analyzes text for scam patterns.
        Returns (list of flags, risk increase amount).
        """
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
        """
        Detects non-lexicon spam/manipulation patterns.
        Returns (list of flags, risk increase amount).
        """
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
