from __future__ import annotations

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
