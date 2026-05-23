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

__all__ = [
    "normalise_arabic",
    "normalise_arabic_cached",
    "contains_arabic_script",
    "extract_arabic_spans",
    "canonicalise_for_matching",
    "is_dominantly_arabic",
]


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
    r"""'"`.,!?():\[\]{}⟨⟩﴾﴿؛٪\u2018\u2019\u201C\u201D\-\u2013\u2014/\\]"""
)

# Alef variants → bare alef (ا)
_ALEF_VARIANTS_RE = re.compile(r"[أإآٱ]")

# Arabic script detection — covers:
#   U+0600–U+06FF  (Arabic block)
#   U+0750–U+077F  (Arabic Supplement)
#   U+FB50–U+FDFF  (Arabic Presentation Forms-A)
#   U+FE70–U+FEFF  (Arabic Presentation Forms-B)
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

# Hamza-on-carrier normalization (O(1) via translate table)
_HAMZA_TABLE = str.maketrans(
    {
        "\u0624": "\u0648",  # ؤ → و
        "\u0626": "\u064a",  # ئ → ي
    }
)

# Taa marbuta + alef maqsura (O(1) via translate table)
_SUFFIX_TABLE = str.maketrans(
    {
        "\u0629": "\u0647",  # ة → ه
        "\u0649": "\u064a",  # ى → ي
    }
)


# ---------------------------------------------------------------------------
# Cross-script canonical mapping (for matching ONLY — never transcript)
# ---------------------------------------------------------------------------
# Whisper outputs Arabic text using Urdu orthography (e.g. اللہ instead of
# الله).  This table maps Urdu-script characters to their Arabic equivalents
# so that the Quran corpus and formula dictionary can match.
#
# SAFETY:
#   - NOT applied to transcript output text — would corrupt Urdu
#   - Only used inside canonicalise_for_matching()
#   - گ (Urdu Gaf) is NOT mapped to غ (Arabic Ghayn) — they are different
#     letters.  Mapping them would cause catastrophic false Quran matches.
#   - ے (Urdu Bari Yeh) is NOT mapped globally — it appears heavily in
#     Urdu grammar (گئے, کیے, چاہیے).  Blind mapping creates false
#     Arabic positives.

_CROSS_SCRIPT_TABLE = str.maketrans(
    {
        "\u06c1": "\u0647",  # ہ (Urdu Heh Goal) → ه (Arabic Heh)
        "\u06cc": "\u064a",  # ی (Urdu Yeh) → ي (Arabic Yeh)
        "\u06a9": "\u0643",  # ک (Urdu Kaf) → ك (Arabic Kaf)
    }
)


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


def canonicalise_for_matching(text: str) -> str:
    """
    Cross-script canonicalization for matching purposes ONLY.

    Converts Urdu-script characters to Arabic equivalents so that
    Whisper's Urdu-orthography Arabic output can match the Quran corpus
    and formula dictionary.

    Pipeline: cross-script table → normalise_arabic()

    NEVER apply this to transcript output text — it will corrupt Urdu.
    Use this ONLY in:
      - QuranMatcher.match() for query canonicalization
      - Formula dictionary key building
      - HadithMatcher.match() for query canonicalization
      - Overlap deduplication comparisons
    """
    if not text:
        return ""
    return normalise_arabic(text.translate(_CROSS_SCRIPT_TABLE))


@lru_cache(maxsize=16384)
def canonicalise_for_matching_cached(text: str) -> str:
    """Cached wrapper around ``canonicalise_for_matching``."""
    return canonicalise_for_matching(text)


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

HALLUCINATION_PHRASES_RAW: frozenset[str] = frozenset(
    {
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
    }
)

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


# ---------------------------------------------------------------------------
# Urdu-aware Arabic span extraction (review 3+4+5 + Colab fix)
# ---------------------------------------------------------------------------

# Urdu-exclusive codepoints: letters used in Urdu but not standard Arabic
_URDU_EXCLUSIVE = frozenset(
    "\u0679\u067e\u0686\u0688\u0691\u0698\u06a9\u06af\u06ba\u06be\u06c1\u06c3\u06cc\u06d2"
)

# Arabic morphology signals that help identify Arabic text even without
# diacritics (Whisper typically strips them). Require BOTH a marker AND
# a morphological signal to avoid false positives on Urdu text.
_ARABIC_DEFINITE_ARTICLE_RE = re.compile(r"\bال\w{2,}")  # ال prefix (al-)
_ARABIC_STOPWORDS = frozenset(
    {
        "من",
        "في",
        "على",
        "الى",
        "عن",
        "ان",
        "ما",
        "لا",
        "الا",
        "هو",
        "هي",
        "هم",
        "الذي",
        "التي",
        "الذين",
        "كل",
        "بعد",
        "قبل",
        "بين",
        "عند",
        "حتى",
        "ثم",
        "او",
        "لم",
        "لن",
        "قد",
    }
)


def extract_arabic_spans(text: str) -> list[str]:
    """
    Extract contiguous Arabic-script spans from mixed Urdu/Arabic text.

    Tokenizes first, then builds spans from consecutive non-Urdu tokens.
    Handles 'Urdu + Quran' as one continuous Arabic-script run without
    discarding the entire run when a single Urdu character appears.

    Acceptance criteria (at least one must be true):
      - Has diacritics (tashkeel) — strong classical Arabic signal
      - Has hamza characters (أ إ ؤ ئ)
      - Has taa marbuta (ة) or alef maqsura (ى)
      - 5+ words (long enough to be meaningful regardless)
      - 3+ words AND Arabic morphology signals (ال prefix + stopwords)
      - 2+ words AND at least 2 Arabic morphology signals
    """
    tokens = text.split()
    spans: list[str] = []
    current_arabic: list[str] = []

    for token in tokens:
        # Check if this token contains Arabic-script characters
        if not any("\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff" for c in token):
            # Non-Arabic token: flush current span if long enough
            if len(current_arabic) >= 2:
                spans.append(" ".join(current_arabic))
            current_arabic = []
            continue

        # Check if token contains Urdu-exclusive characters
        if any(c in _URDU_EXCLUSIVE for c in token):
            # Urdu token: flush current span
            if len(current_arabic) >= 2:
                spans.append(" ".join(current_arabic))
            current_arabic = []
        else:
            current_arabic.append(token)

    # Flush remaining
    if len(current_arabic) >= 2:
        spans.append(" ".join(current_arabic))

    # Filter: require Arabic signals to accept the span
    filtered: list[str] = []
    for span in spans:
        has_diacritics = bool(re.search(r"[\u064B-\u065F\u0670]", span))
        has_hamza = bool(re.search(r"[\u0624\u0626\u0623\u0625]", span))
        has_classical = bool(re.search(r"[\u0629\u0649]", span))
        word_count = len(span.split())

        # Strong signals: always accept
        if has_diacritics or has_hamza or has_classical or word_count >= 5:
            filtered.append(span)
            continue

        # Medium signals: Arabic morphology (ال prefix + stopwords)
        span_words = set(span.split())
        al_count = len(_ARABIC_DEFINITE_ARTICLE_RE.findall(span))
        stopword_hits = len(span_words & _ARABIC_STOPWORDS)
        morphology_signals = al_count + stopword_hits

        if (
            word_count >= 3
            and morphology_signals >= 1
            or word_count >= 2
            and morphology_signals >= 2
        ):
            filtered.append(span)

    return filtered


def is_dominantly_arabic(text: str) -> bool:
    """True when Arabic-script characters dominate and Urdu signals are weak."""
    if not text:
        return False
    arabic_chars = 0
    urdu_exclusive_chars = 0
    total_script_chars = 0
    for c in text:
        if "\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff":
            total_script_chars += 1
            if c in _URDU_EXCLUSIVE:
                urdu_exclusive_chars += 1
            else:
                arabic_chars += 1
        elif c.isalpha():
            total_script_chars += 1
    if total_script_chars == 0:
        return False
    arabic_ratio = arabic_chars / total_script_chars
    urdu_ratio = urdu_exclusive_chars / max(arabic_chars + urdu_exclusive_chars, 1)
    return arabic_ratio > 0.60 and urdu_ratio < 0.20
