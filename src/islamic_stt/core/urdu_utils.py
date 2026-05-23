"""
urdu_utils.py
-------------
Urdu text normalization for the Islamic STT pipeline.

Safe normalizations only — no linguistically unsafe rules.

Rules:
  - Arabic Yeh variants → Urdu Yeh (ي → ی, ى → ی)
  - Arabic Kaf → Urdu Kaf (ك → ک)
  - ZWNJ artifact removal
  - Unicode presentation form cleanup
  - Urdu punctuation normalization
  - Roman Urdu → Urdu script (phrase-level, Islamic context only)

IMPORTANT: ے (Urdu Bari Ye) is NOT normalized to ی.
That mapping is linguistically unsafe and breaks Urdu grammar
(گئے, کیے, چاہیے would become incorrect).
"""

from __future__ import annotations

import re

__all__ = ["normalise_urdu", "roman_urdu_to_script"]


# ---------------------------------------------------------------------------
# Character-level normalization tables
# ---------------------------------------------------------------------------

# Safe Arabic→Urdu character mappings
_CHAR_TABLE = str.maketrans(
    {
        "\u064a": "\u06cc",  # ي (Arabic Yeh) → ی (Urdu Yeh)
        "\u0649": "\u06cc",  # ى (Alef Maqsura) → ی (Urdu Yeh)
        "\u0643": "\u06a9",  # ك (Arabic Kaf) → ک (Urdu Kaf)
        "\u0629": "\u06c3",  # ة (Taa Marbuta) → ۃ (Urdu Taa Marbuta)
    }
)

# ZWNJ and related invisible characters
_INVISIBLE_RE = re.compile(r"[\u200C\u200D\u200E\u200F\u202A-\u202E\uFEFF]")

# Collapse multiple spaces
_MULTISPACE_RE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalise_urdu(text: str) -> str:
    """
    Normalize Urdu text with safe character mappings.

    Does NOT touch ے (Bari Ye) — that mapping breaks Urdu grammar.
    """
    if not text:
        return text

    # Character-level normalization
    text = text.translate(_CHAR_TABLE)

    # Remove ZWNJ and other invisible formatting
    text = _INVISIBLE_RE.sub("", text)

    # Collapse whitespace
    text = _MULTISPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Roman Urdu → Urdu script (phrase-level, Islamic context only)
# ---------------------------------------------------------------------------

# Only Islamic phrases — no single-word mappings to avoid false positives.
# Keys must be lowercase. Only matches full phrase boundaries.
_ROMAN_URDU_PHRASES: dict[str, str] = {
    # Hadith collections
    "sahih bukhari": "صحیح بخاری",
    "sahih muslim": "صحیح مسلم",
    "sunan tirmidhi": "سنن ترمذی",
    "sunan abu dawud": "سنن ابو داؤد",
    "sunan nasai": "سنن نسائی",
    "sunan ibn majah": "سنن ابن ماجہ",
    # Quran-related
    "surah fatiha": "سورۃ فاتحہ",
    "surah baqarah": "سورۃ بقرہ",
    "surah yasin": "سورۃ یٰس",
    "surah rahman": "سورۃ رحمٰن",
    "surah mulk": "سورۃ ملک",
    "surah kahf": "سورۃ کہف",
    "ayatul kursi": "آیت الکرسی",
    # Islamic terms
    "peace be upon him": "صلی اللہ علیہ وسلم",
    "pbuh": "صلی اللہ علیہ وسلم",
    "insha allah": "ان شاء اللہ",
    "masha allah": "ما شاء اللہ",
    "subhan allah": "سبحان اللہ",
    "alhamdulillah": "الحمد للہ",
    "allahu akbar": "اللہ اکبر",
    "assalamu alaikum": "السلام علیکم",
    "jazakallah": "جزاک اللہ",
    "bismillah": "بسم اللہ",
    # Scholar names (phrase-level only)
    "imam bukhari": "امام بخاری",
    "imam muslim": "امام مسلم",
    "imam abu hanifa": "امام ابو حنیفہ",
    "ibn taymiyyah": "ابن تیمیہ",
    "ibn kathir": "ابن کثیر",
}

# Pre-compile patterns (word-boundary aware, case-insensitive)
_ROMAN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(eng) + r"\b", re.IGNORECASE), urdu)
    for eng, urdu in _ROMAN_URDU_PHRASES.items()
]


def roman_urdu_to_script(text: str) -> str:
    """
    Replace known Roman Urdu Islamic phrases with Urdu script equivalents.

    Only replaces full phrase matches (word-boundary aware).
    Single words like 'ali', 'noor' are NOT replaced to avoid false positives.
    """
    for pattern, replacement in _ROMAN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
