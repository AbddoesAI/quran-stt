"""
arabic_utils.py
---------------
Shared Arabic text canonicalization utilities used by all matchers
and the language detector.

Normalization pipeline
----------------------
1.  Unicode NFC normalization
2.  Strip tashkeel (diacritics / harakat) and tatweel (kashida)
3.  Normalize alef variants (أ إ آ ٱ → ا)
4.  Normalize hamza-on-carrier (ؤ → و, ئ → ي)
5.  Normalize taa marbuta (ة → ه) and alef maqsura (ى → ي)
6.  Strip Arabic and Latin punctuation + bidi control characters
7.  Collapse whitespace

This module is intentionally dependency-free (stdlib only) so that
every other module can import it without circular-import risk.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

__all__ = ["normalise_arabic", "normalise_arabic_cached", "contains_arabic_script"]


# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Tashkeel (harakat / diacritics) + tatweel (kashida U+0640)
# Covers:  U+0610–U+061A  (Quranic annotation signs)
#          U+064B–U+065F  (standard Arabic diacritics)
#          U+0670         (superscript alef)
#          U+06D6–U+06DC  (Quranic marks)
#          U+06DF–U+06E4  (Quranic marks continued)
#          U+06E7–U+06E8  (small high letters)
#          U+06EA–U+06ED  (Quranic tone marks)
#          U+0640         (tatweel / kashida)
_TASHKEEL_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC"
    r"\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]"
)

# Arabic + Latin punctuation + Unicode bidi controls + zero-width chars
_PUNCT_RE = re.compile(
    r"[،؛؟«»\u200F\u200E\u200B\u200C\u200D\uFEFF"
    r'""\"\'`.,!?():\[\]{}⟨⟩﴾﴿؛٪\-–—/\\]'
)

# Alef variants → bare alef (ا)
_ALEF_VARIANTS_RE = re.compile(r"[أإآٱ]")

# Arabic script detection — covers:
#   U+0600–U+06FF  (Arabic block)
#   U+0750–U+077F  (Arabic Supplement)
#   U+FB50–U+FDFF  (Arabic Presentation Forms-A)
#   U+FE70–U+FEFF  (Arabic Presentation Forms-B)
_ARABIC_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Hamza-on-carrier normalization (O(1) via translate table)
_HAMZA_TABLE = str.maketrans({
    "\u0624": "\u0648",  # ؤ → و
    "\u0626": "\u064A",  # ئ → ي
})

# Taa marbuta + alef maqsura (O(1) via translate table)
_SUFFIX_TABLE = str.maketrans({
    "\u0629": "\u0647",  # ة → ه
    "\u0649": "\u064A",  # ى → ي
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def contains_arabic_script(text: str) -> bool:
    """Return True if *text* contains at least one Arabic-script character."""
    return bool(_ARABIC_SCRIPT_RE.search(text or ""))


def normalise_arabic(text: str) -> str:
    """
    Canonical normalization for fuzzy matching.

    Deterministic and idempotent — calling it twice yields the same result.
    Intentionally aggressive: strips ALL diacritics and normalizes letter
    variants so that Whisper output (which is inconsistent with harakat)
    can be compared against the fully-diacritized Quran corpus.

    Parameters
    ----------
    text : Raw Arabic/Urdu/mixed text.

    Returns
    -------
    Normalized string with collapsed whitespace, no diacritics, and
    canonical letter forms.
    """
    if not text:
        return ""
    # 1. Unicode NFC (compose combining sequences)
    text = unicodedata.normalize("NFC", text)
    # 2. Strip tashkeel + tatweel
    text = _TASHKEEL_RE.sub("", text)
    # 3. Normalize alef variants
    text = _ALEF_VARIANTS_RE.sub("\u0627", text)  # → ا
    # 4. Normalize hamza carriers
    text = text.translate(_HAMZA_TABLE)
    # 5. Normalize taa marbuta + alef maqsura
    text = text.translate(_SUFFIX_TABLE)
    # 6. Strip punctuation + bidi controls
    text = _PUNCT_RE.sub(" ", text)
    # 7. Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


@lru_cache(maxsize=16384)
def normalise_arabic_cached(text: str) -> str:
    """
    Cached wrapper around ``normalise_arabic``.

    Use this in hot loops where the same text may be normalized multiple
    times (e.g. corpus loading, repeated matcher calls).  The cache holds
    up to 16384 entries (~2.5× the Quran corpus size) so the full corpus
    fits without eviction.

    Do NOT use for one-shot normalization of user input — the cache
    overhead is not worth it for single calls.
    """
    return normalise_arabic(text)


# ---------------------------------------------------------------------------
# Hallucination blocklist (single source of truth)
# ---------------------------------------------------------------------------
# Known Whisper hallucination phrases from YouTube-trained models.
# Used by both transcriber.py and quran_matcher.py — defined here to
# avoid duplication drift.
#
# Expanded from 7 → 35+ phrases based on observed Whisper large-v3
# hallucination patterns in Arabic/Urdu/English audio.

HALLUCINATION_PHRASES_RAW: frozenset[str] = frozenset({
    # --- Arabic YouTube-trained hallucinations ---
    "اشتركوا في القناة",
    "شكرا للمشاهدة",
    "لا تنسوا الاشتراك",
    "اشترك في القناة",
    "شكراً للمشاهدة",
    "لا تنسى الاشتراك والإعجاب",
    "تابعونا على",
    "اشتراك في القناة",
    "لا تنسى الاشتراك في القناة",
    "اشترك وفعل زر الجرس",
    "اذا اعجبك الفيديو",
    "لا تنسى الاشتراك",
    "شكرا لكم على المشاهدة",
    "مشاهدة ممتعة",
    "نراكم في الحلقة القادمة",
    "السلام عليكم ورحمة الله",  # only when isolated (not in context)
    # --- English YouTube hallucinations ---
    "Thanks for watching",
    "Please subscribe",
    "Don't forget to subscribe",
    "Like and subscribe",
    "Hit the bell icon",
    "See you in the next video",
    "Thank you for watching",
    "Please like and subscribe",
    # --- Urdu YouTube hallucinations ---
    "چینل کو سبسکرائب کریں",
    "لائک اور سبسکرائب کریں",
    "ویڈیو کو لائک کریں",
    # --- Whisper silence/noise hallucinations ---
    "...",
    "♪",
    "♪♪",
    "♪♪♪",
    "[موسيقى]",
    "[تصفيق]",
    "[音楽]",
    "MBC",
    "Amara.org",
    "www.mooji.org",
    "Sous-titres réalisés par la communauté",
    "ترجمة",
    "Subtítulos",
})

HALLUCINATION_PHRASES_NORMALISED: frozenset[str] = frozenset(
    normalise_arabic(p) for p in HALLUCINATION_PHRASES_RAW
)

# Patterns that indicate hallucination by structure (not exact match)
HALLUCINATION_PATTERNS: list[re.Pattern] = [
    # Repeated single word 8+ times (raised from 4 to protect dhikr).
    # Exclude known dhikr words from this check.
    re.compile(
        r"^(?!.*(الله|سبحان|الحمد|استغفر|اكبر))"  # negative lookahead: skip dhikr
        r"(\S+)(\s+\2){7,}$"
    ),
    # Very short segment that is just punctuation or whitespace
    re.compile(r"^\s*[\.…,،]+\s*$"),
    # Music/sound markers
    re.compile(r"^\s*[♪♫🎵🎶]+\s*$"),
    # URL-like content
    re.compile(r"https?://|www\.|\.com|\.org"),
]
