from __future__ import annotations

class SafetyService:
    """
    Service for detecting manipulation, spam, and slang-based scams.
    """
    SCAM_LEXICON = {
        "guaranteed returns",
        "jackpot tips",
        "double your investment",
        "pump and dump",
        "insider info",
        "100% success rate",
        "no risk",
        "get rich quick",
    }

    def check_for_scams(self, text: str) -> tuple[list[str], float]:
        """
        Analyzes text for scam patterns.
        Returns (list of flags, risk increase amount).
        """
        text_lower = text.lower()
        flags = []
        risk_increase = 0.0

        for pattern in self.SCAM_LEXICON:
            if pattern in text_lower:
                flags.append(f"scam_pattern:{pattern.replace(' ', '_')}")
                risk_increase += 0.25
        
        return flags, min(risk_increase, 1.0)
