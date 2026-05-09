"""
language_detector.py
--------------------
Per-segment language detection.

Strategy (layered, fast → accurate):
1.  Trust Whisper's own language field when it is present and the segment
    is long enough to be reliable (≥ 3 words).
2.  For short segments or when Whisper's label is absent/ambiguous, run
    langdetect as a fast first pass.
3.  If langdetect returns low confidence or ambiguity between Arabic and
    Urdu (a common problem given shared script characters), fall back to
    Lingua which has stronger Arabic/Urdu/Farsi discrimination.

Output is always a normalised ISO-639-1 code:
    'ar'  → Arabic
    'ur'  → Urdu
    'en'  → English
    'und' → undetermined

Arabic is declared only when genuinely Arabic-script, non-Urdu text is
detected so that Urdu segments are not incorrectly routed to the Quran
matcher.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from islamic_stt.core.arabic_utils import normalise_arabic

logger = logging.getLogger(__name__)

# langdetect — seed for deterministic results across runs (MED-5 fix)
from langdetect import detect_langs, LangDetectException
from langdetect import DetectorFactory
DetectorFactory.seed = 0

__all__ = ["detect_language"]

# Lingua (higher accuracy, heavier)
try:
    from lingua import Language, LanguageDetectorBuilder

    _LINGUA_DETECTOR = (
        LanguageDetectorBuilder.from_languages(
            Language.ARABIC,
            Language.URDU,
            Language.ENGLISH,
        )
        .with_minimum_relative_distance(0.15)
        .build()
    )
    _LINGUA_AVAILABLE = True
except ImportError:
    _LINGUA_AVAILABLE = False
    _LINGUA_DETECTOR = None  # sentinel — prevents NameError in _run_lingua()
    logger.warning(
        "lingua not available — falling back to langdetect only. "
        "Install via: pip install lingua-language-detector"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum word count for Whisper's own label to be trusted without
# secondary verification.
_WHISPER_TRUST_MIN_WORDS = 3

# langdetect confidence threshold below which we invoke Lingua.
_LANGDETECT_CONFIDENCE_THRESHOLD = 0.85

# Map Lingua Language enum → ISO code.
_LINGUA_TO_ISO: dict[str, str] = {
    "ARABIC": "ar",
    "URDU": "ur",
    "ENGLISH": "en",
}

# Arabic Unicode block range: U+0600 – U+06FF
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")

# High-frequency Urdu function words / particles that Whisper outputs in
# standard Arabic characters (without Urdu-exclusive code points).  These
# words almost never appear in Classical or Quranic Arabic.
# The list is deliberately broad to catch conversational Urdu lectures.
_URDU_VOCAB: frozenset[str] = frozenset({
    # Pronouns & demonstratives
    "ہے", "ہیں", "ہو", "ہوں", "ہوگا", "ہوگی",
    "یہ", "وہ", "کیا", "کون", "کہاں", "کب", "کیسے", "کیوں",
    # Postpositions / particles
    "کی", "کا", "کو", "کے", "سے", "میں", "پر", "نے", "تک",
    # Verbs / auxiliaries
    "کرنا", "کرتا", "کرتے", "کریں", "رہا", "رہے", "رہی",
    "ہوتا", "ہوتی", "ہوتے", "جاتا", "جاتے", "آتا", "آتے",
    # Common conversational words
    "لوگ", "لوگوں", "بہت", "ابھی", "بھی", "نہیں", "ہاں",
    "لیکن", "مگر", "اور", "پھر", "تو", "جب", "اگر",
    "والا", "والے", "والی", "اچھا", "اچھے", "بات",
    # Whisper-romanised Urdu (Arabic script, no Urdu codepoints)
    "هوتا", "هوتي", "هونا", "كرنا", "كرتا", "كرتي",
    "لوگ", "بهت", "نهيں", "كيا", "كيوں", "ليكن",
    "هاں", "بهي", "ابهي", "پهر", "اچها",
    # Mixed-script tokens observed in real Whisper output
    "خواهش", "دروازة", "روتين", "باكستان",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(segment_text: str, whisper_language: Optional[str] = None) -> str:
    """
    Return the most likely ISO-639-1 language code for *segment_text*.

    Parameters
    ----------
    segment_text     : The raw transcribed text of one segment.
    whisper_language : The language code Whisper already assigned, if any.

    Returns
    -------
    ISO-639-1 code ('ar', 'ur', 'en', or 'und').
    """
    text = segment_text.strip()

    if not text:
        return "und"

    word_count = len(text.split())

    # --- Step 1: trust Whisper for sufficiently long segments ---------------
    if whisper_language and word_count >= _WHISPER_TRUST_MIN_WORDS:
        normalised = _normalise_whisper_code(whisper_language)
        # Even if Whisper says 'ar', verify it's not Urdu
        if normalised == "ar" and _is_likely_urdu(text):
            normalised = "ur"
        if normalised != "und":
            return normalised

    # --- Step 2: langdetect fast pass ---------------------------------------
    langdetect_result = _run_langdetect(text)
    if langdetect_result["confidence"] >= _LANGDETECT_CONFIDENCE_THRESHOLD:
        code = langdetect_result["language"]
        # Resolve Arabic vs Urdu ambiguity
        if code in ("ar", "ur"):
            code = _resolve_arabic_urdu(text)
        return code

    # --- Step 2b: Lingua as primary for Arabic-script text (REC-1) ---------
    # langdetect was low-confidence.  For Arabic-script text, Lingua is far
    # better at separating Arabic from Urdu, so promote it to primary here
    # rather than using it only as a last-resort fallback.
    if _LINGUA_AVAILABLE and _ARABIC_SCRIPT_RE.search(text):
        lingua_code = _run_lingua(text)
        if lingua_code != "und":
            return lingua_code

    # --- Step 3: Lingua fallback for non-Arabic-script text ----------------
    if _LINGUA_AVAILABLE:
        lingua_code = _run_lingua(text)
        if lingua_code != "und":
            return lingua_code

    # Return whatever langdetect gave us (best effort)
    return langdetect_result["language"] or "und"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_whisper_code(code: str) -> str:
    """Map Whisper language codes to our canonical ISO-639-1 set."""
    mapping = {
        "arabic": "ar",
        "ar": "ar",
        "urdu": "ur",
        "ur": "ur",
        "english": "en",
        "en": "en",
    }
    return mapping.get(code.lower(), "und")


def _run_langdetect(text: str) -> dict:
    """
    Run langdetect and return {'language': str, 'confidence': float}.
    Returns 'und' / 0.0 on failure.
    """
    try:
        probs = detect_langs(text)
        # probs is a list of Lang objects sorted by probability (desc)
        top = probs[0]
        lang = top.lang
        # Normalise codes: langdetect uses 'zh-cn' style; we only care about base
        lang = lang.split("-")[0]
        return {"language": lang, "confidence": float(top.prob)}
    except LangDetectException:
        return {"language": "und", "confidence": 0.0}


def _run_lingua(text: str) -> str:
    """Run Lingua and return an ISO-639-1 code, or 'und'."""
    if _LINGUA_DETECTOR is None:
        return "und"
    result = _LINGUA_DETECTOR.detect_language_of(text)
    if result is None:
        return "und"
    return _LINGUA_TO_ISO.get(result.name, "und")


def _resolve_arabic_urdu(text: str) -> str:
    """
    Both Arabic and Urdu share the Arabic script so surface-level detectors
    frequently confuse them.  We use multiple tiebreakers:
      1. Urdu-exclusive Unicode characters (ٹ ڈ ڑ etc.)
      2. Urdu vocabulary words (common function words / particles)
      3. Lingua classifier (if available)

    Returns 'ur' if Urdu signals are found, else 'ar'.
    """
    if _is_likely_urdu(text):
        return "ur"
    # Last resort: ask Lingua for Arabic-script disambiguation
    if _LINGUA_AVAILABLE:
        lingua_code = _run_lingua(text)
        if lingua_code in ("ar", "ur"):
            return lingua_code
    return "ar"


# Pre-compiled regex for Urdu-exclusive characters.
_URDU_EXCLUSIVE_RE = re.compile(r"[\u0679\u0688\u0691\u06BA\u06BE\u06D2\u06D3\u06D4]")


def _is_likely_urdu(text: str) -> bool:
    """
    True if the text contains Urdu signals that distinguish it from Arabic.

    Two detection layers (REC-1 fix):
      1. Urdu-exclusive Unicode characters (ٹ ڈ ڑ ں ھ ے ۓ ۔) — these
         never appear in Classical Arabic.
      2. Urdu vocabulary words — high-frequency function words like
         کی، ہے، نہیں، بہت etc. that Whisper often outputs in standard
         Arabic characters (without Urdu-exclusive codepoints).
    """
    # Layer 1: Urdu-exclusive characters
    if _URDU_EXCLUSIVE_RE.search(text):
        return True

    # Layer 2: Urdu vocabulary (matches ≥ 2 words to avoid false positives)
    words = set(normalise_arabic(text).split())
    urdu_hits = words & _URDU_VOCAB
    if len(urdu_hits) >= 2:
        return True

    return False
