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

HALLUCINATION_PHRASES_RAW: frozenset[str] = frozenset({
    "اشتركوا في القناة",
    "شكرا للمشاهدة",
    "لا تنسوا الاشتراك",
    "اشترك في القناة",
    "شكراً للمشاهدة",
    "لا تنسى الاشتراك والإعجاب",
    "تابعونا على",
})

HALLUCINATION_PHRASES_NORMALISED: frozenset[str] = frozenset(
    normalise_arabic(p) for p in HALLUCINATION_PHRASES_RAW
)
