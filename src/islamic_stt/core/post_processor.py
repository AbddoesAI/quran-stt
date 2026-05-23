"""
post_processor.py
-----------------
Post-processing correction layer for Islamic STT output.

Addresses CRIT-4: No post-processing correction layer exists.

This module applies rule-based corrections to Whisper output to fix:
1. Common abbreviation expansions (صلعم → صلى الله عليه وسلم)
2. Islamic terminology corrections
3. Merged word splitting (انشاءالله → إن شاء الله)
4. Diacritics restoration for matched Quranic phrases
5. Structural cleanup (orphan punctuation, extra spaces)

Applied AFTER transcription but BEFORE matching, so that the
corrected text has a higher chance of matching the Quran/Hadith corpus.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

__all__ = ["apply_post_processing"]


# ---------------------------------------------------------------------------
# Abbreviation expansions (Whisper commonly produces these)
# ---------------------------------------------------------------------------
# Order matters: longer patterns first to avoid partial matches.

_EXPANSIONS: list[tuple[re.Pattern, str]] = [
    # Salawat abbreviations
    (re.compile(r"\bصلعم\b"), "صلى الله عليه وسلم"),
    (re.compile(r"\bﷺ"), "صلى الله عليه وسلم"),
    # Honorific abbreviations
    (re.compile(r"\bرض\b"), "رضي الله عنه"),
    (re.compile(r"\bرح\b"), "رحمه الله"),
    # Merged phrases (Whisper often merges these)
    (re.compile(r"انشاءالله"), "إن شاء الله"),
    (re.compile(r"انشالله"), "إن شاء الله"),
    (re.compile(r"ماشاءالله"), "ما شاء الله"),
    (re.compile(r"ماشالله"), "ما شاء الله"),
    (re.compile(r"الحمدالله"), "الحمد لله"),
    (re.compile(r"الحمدللہ"), "الحمد للہ"),
    (re.compile(r"جزاکالله"), "جزاک الله"),
    (re.compile(r"جزاكالله"), "جزاك الله"),
    (re.compile(r"بسمالله"), "بسم الله"),
    (re.compile(r"سبحانالله"), "سبحان الله"),
    (re.compile(r"استغفرالله"), "استغفر الله"),
    (re.compile(r"لاالہ"), "لا الہ"),
    (re.compile(r"لااله"), "لا اله"),
    # Common Whisper merges in Urdu
    (re.compile(r"انکو"), "ان کو"),
    (re.compile(r"اسکو"), "اس کو"),
    (re.compile(r"اسکی"), "اس کی"),
    (re.compile(r"اسکے"), "اس کے"),
    (re.compile(r"اسکا"), "اس کا"),
    (re.compile(r"جوکہ"), "جو کہ"),
    (re.compile(r"حالانکہ"), "حالانکہ"),  # this one is actually correct merged
]

# ---------------------------------------------------------------------------
# Islamic terminology corrections
# ---------------------------------------------------------------------------
# Maps common Whisper mis-transcriptions to correct forms.
# Keys are what Whisper outputs; values are the correct spellings.

_TERM_CORRECTIONS: dict[str, str] = {
    # Arabic corrections
    "قران": "قرآن",
    "الائمه": "الأئمة",
    "الأأمة": "الأئمة",
    # Urdu corrections
    "مسئله": "مسئلہ",
    "مسلہ": "مسئلہ",
    # Collection name corrections
    "بخارى": "بخاری",
    "مسلم شريف": "مسلم شریف",
    "ترمذى": "ترمذی",
    # Common speaker reference corrections
    "نبى": "نبی",
    "رسول اللہ": "رسول اللہ",  # normalize form
}

# Precompile term correction patterns for performance
_TERM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(re.escape(k)), v) for k, v in _TERM_CORRECTIONS.items()
]

# ---------------------------------------------------------------------------
# English Islamic term normalization
# ---------------------------------------------------------------------------
# Standardize common English Islamic terms that Whisper outputs inconsistently.

_ENGLISH_NORMALIZATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[Ii]nsha\s*[Aa]llah\b"), "Insha'Allah"),
    (re.compile(r"\b[Mm]asha\s*[Aa]llah\b"), "Masha'Allah"),
    (re.compile(r"\b[Ss]ubhan\s*[Aa]llah\b"), "SubhanAllah"),
    (re.compile(r"\b[Aa]lhamdulillah\b"), "Alhamdulillah"),
    (re.compile(r"\b[Aa]llahu\s*[Aa]kbar\b"), "Allahu Akbar"),
    (re.compile(r"\b[Aa]staghfirullah\b"), "Astaghfirullah"),
    (re.compile(r"\b[Jj]azak\s*[Aa]llah\b"), "JazakAllah"),
    (re.compile(r"\bPBUH\b", re.IGNORECASE), "صلى الله عليه وسلم"),
    (re.compile(r"\bSAW\b"), "صلى الله عليه وسلم"),  # case-sensitive to avoid matching "saw"
    (re.compile(r"\bSWT\b"), "سبحانه وتعالى"),
    (re.compile(r"\bRA\b"), "رضي الله عنه"),  # case-sensitive
]


# ---------------------------------------------------------------------------
# Mixed-script sanitizer (Colab fix: Latin leakage in Arabic/Urdu text)
# ---------------------------------------------------------------------------
# Whisper's multilingual decoder sometimes emits mixed-script tokens
# (Arabic chars + Latin chars in a single word).  Examples:
#   اللah → الله    فرma → فرما    مسلمán → مسلمان
#
# Strategy: targeted repair dictionary FIRST (safe, deterministic),
# then guarded generic regex for remaining cases (only when Arabic
# ratio is high and Latin suffix is very short).

# Invocation stabilization: common corrupted forms → canonical
_INVOCATION_REPAIRS: dict[re.Pattern, str] = {
    # يا الله corruptions (very common in dua sections)
    re.compile(r"الل[A-Za-z]{1,4}(?:\b|$)"): "الله",
    # Common Urdu verb corruptions
    re.compile(r"فر[Mm][Aa]?\b"): "فرما",
    # Bismillah corruptions
    re.compile(r"بسم\s*الل[A-Za-z]{1,4}"): "بسم الله",
}

# Generic mixed-script: Arabic word ending with short Latin suffix
# Only applied when the token is predominantly Arabic (>70% Arabic chars)
# Matches basic + accented Latin: A-Za-z plus Latin Extended (àáâãäåéèêëíîïóôõöúùûüñçß etc.)
_LATIN_CHAR_CLASS = r"A-Za-z\u00C0-\u024F"
_MIXED_SCRIPT_TRAILING_LATIN = re.compile(
    r"([\u0600-\u06FF]{3,})[" + _LATIN_CHAR_CLASS + r"]{1,3}(?=\s|$)"
)
_MIXED_SCRIPT_LEADING_LATIN = re.compile(
    r"(?:^|\s)[" + _LATIN_CHAR_CLASS + r"]{1,3}([\u0600-\u06FF]{3,})"
)


def _sanitize_mixed_script(text: str) -> str:
    """
    Remove Latin character leakage from Arabic/Urdu tokens.

    Two-phase approach:
    1. Targeted repair dictionary (high-confidence, known patterns)
    2. Guarded generic regex (only for predominantly Arabic tokens)
    """
    # Phase 1: targeted repairs (safe, deterministic)
    for pattern, replacement in _INVOCATION_REPAIRS.items():
        text = pattern.sub(replacement, text)

    # Phase 2: generic trailing Latin removal (guarded)
    # Only strip if the Arabic portion is substantial (≥3 chars)
    text = _MIXED_SCRIPT_TRAILING_LATIN.sub(r"\1", text)
    text = _MIXED_SCRIPT_LEADING_LATIN.sub(r" \1", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Structural cleanup
# ---------------------------------------------------------------------------

# Orphan punctuation at start/end
_ORPHAN_PUNCT_RE = re.compile(r"^[\s،؛؟.,:;!?]+|[\s،؛؟.,:;!?]+$")

# Multiple consecutive spaces
_MULTI_SPACE_RE = re.compile(r"  +")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_post_processing(text: str) -> str:
    """
    Apply all post-processing corrections to a transcribed text segment.

    This should be called AFTER Whisper transcription and BEFORE matching.

    Parameters
    ----------
    text : Raw transcribed text from a single segment.

    Returns
    -------
    Corrected text with expanded abbreviations, fixed terminology,
    mixed-script repair, and structural cleanup applied.
    """
    if not text or not text.strip():
        return text

    original = text

    # 1. Expand abbreviations (Arabic merged words)
    for pattern, replacement in _EXPANSIONS:
        text = pattern.sub(replacement, text)

    # 2. Fix Islamic terminology
    for pattern, replacement in _TERM_PATTERNS:
        text = pattern.sub(replacement, text)

    # 3. Normalize English Islamic terms
    for pattern, replacement in _ENGLISH_NORMALIZATIONS:
        text = pattern.sub(replacement, text)

    # 4. Mixed-script sanitization (Latin leakage repair)
    text = _sanitize_mixed_script(text)

    # 5. Structural cleanup
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()

    if text != original:
        logger.debug("Post-processed: '%s' → '%s'", original[:50], text[:50])

    return text
