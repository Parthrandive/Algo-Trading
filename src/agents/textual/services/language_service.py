from __future__ import annotations
import re

class LanguageService:
    """
    Service for detecting and normalizing Hinglish/Code-mixed text.
    """
    HINGLISH_KEYWORDS = {
        "bhai", "paisa", "market up jayega", "nifty girega", 
        "kya lagta hai", "profit hoga", "call le lo", "put buy karo"
    }

    def detect_language(self, text: str) -> str:
        """Returns 'en', 'hi', or 'code_mixed'."""
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.HINGLISH_KEYWORDS):
            return "code_mixed"
        
        # Simple heuristic for romanized hindi patterns (repeating vowels, common endings)
        if re.search(r"\b(hai|haii|tha|the|rha|rhi|ja|aa)\b", text_lower):
            return "code_mixed"
            
        return "en"

    def normalize_hinglish(self, text: str) -> str:
        """Basic normalization for common market Hinglish slang."""
        normalized = text.lower()
        # Example translations/normalizations
        substitutions = {
            "paisa": "money",
            "girega": "will fall",
            "jayega": "will go",
        }
        for slang, norm in substitutions.items():
            normalized = normalized.replace(slang, norm)
        
        return normalized.strip()
