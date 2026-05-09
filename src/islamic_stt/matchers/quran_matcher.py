"""
quran_matcher.py
----------------
Matches a transcribed Arabic segment against a local Quran corpus loaded
from data/quran.json.

Expected quran.json schema
--------------------------
The file should follow the quran-json project format:
https://github.com/risan/quran-json

Top-level structure:
[
  {
    "id": 1,          ← surah number
    "name": "Al-Fatihah",
    "verses": [
      {
        "id": 1,      ← ayah number within surah
        "text": "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
      },
      ...
    ]
  },
  ...
]

Matching strategy
-----------------
1.  Normalise both the query and the corpus (strip diacritics / tashkeel,
    collapse whitespace, remove tatweel).
2.  Exact match on the normalised string → high confidence (1.0).
3.  Rapid fuzzy match using rapidfuzz.fuzz.partial_ratio →
    returns a score 0–100; we expose it as 0.0–1.0.
4.  Returns None when the best score is below FUZZY_THRESHOLD.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from rapidfuzz import fuzz, process as rf_process
from islamic_stt.core.arabic_utils import normalise_arabic, HALLUCINATION_PHRASES_RAW, HALLUCINATION_PHRASES_NORMALISED

logger = logging.getLogger(__name__)


__all__ = ["QuranMatcher", "QuranMatch", "FormulaMatch"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Confidence threshold to accept a fuzzy match (0.0 – 1.0).
# 0.88 was tuned on real lecture output to reject false positives on common
# short phrases like "هذا الحق" and "يا رسول الله".  (REC-2 fix)
FUZZY_THRESHOLD = 0.88

# Minimum character length of the normalised query for fuzzy matching.
# Short queries produce too many false positives against short ayahs.
MIN_FUZZY_QUERY_CHARS = 20
MIN_EXACT_MATCH_CHARS = 12

# Default corpus path (relative to CWD — the pipeline always passes
# config.quran_corpus explicitly, so this is only a fallback for direct usage).
_DEFAULT_CORPUS_PATH = os.path.join("data", "quran.json")


# ---------------------------------------------------------------------------
# Normalisation  (must be defined before KNOWN_ISLAMIC_FORMULAS and corpus loader)
# ---------------------------------------------------------------------------

# Regex for Arabic diacritics (tashkeel) and decorative characters.
def _normalise_arabic(text: str) -> str:
    """Backward-compatible shim for old imports."""
    return normalise_arabic(text)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QuranMatch:
    surah_id: int
    surah_name: str
    ayah_id: int
    original_text: str      # from corpus, with full diacritics
    matched_text: str       # what the transcription contained
    confidence: float       # 0.0 – 1.0
    is_exact: bool


@dataclass
class FormulaMatch:
    """A recognised Islamic formula (Salawat, honorific, dhikr, etc.)."""
    label: str              # human-readable name, e.g. "Salawat (Durood Ibrahim)"
    matched_text: str       # the raw transcription text that triggered this
    confidence: float       # always 1.0 for exact normalised match


# ---------------------------------------------------------------------------
# Known Islamic Formulas
# ---------------------------------------------------------------------------
# Keys are the *normalised* Arabic (no diacritics, collapsed whitespace) so
# they match Whisper output directly.  Values are descriptive labels.

KNOWN_ISLAMIC_FORMULAS: dict[str, str] = {
    _normalise_arabic(k): v
    for k, v in {
        # --- Salawat & honorifics ---
        "صلى الله عليه وسلم":    "Salawat (Durood Ibrahim)",
        "رضي الله عنه":          "Radhi Allahu anhu",
        "رضي الله عنها":         "Radhi Allahu anha",
        "رضي الله عنهم":         "Radhi Allahu anhum",
        "رحمه الله":             "Rahimahullah",
        "رحمها الله":            "Rahimahallah",
        # --- Names of Allah ---
        "جل جلاله":              "Jalla Jalaluhu",
        "سبحانه وتعالى":         "Subhanahu wa Ta'ala",
        "عز وجل":                "Azza wa Jall",
        "تبارك وتعالى":          "Tabaraka wa Ta'ala",
        # --- Dhikr ---
        "الله اكبر":              "Takbir",
        "سبحان الله":             "Tasbih",
        "الحمد لله":              "Tahmid",
        "لا اله الا الله":         "Tahlil",
        "استغفر الله":            "Istighfar",
        "لا حول ولا قوة الا بالله": "Hawqala",
        "انا لله وانا اليه راجعون": "Istirja (Inna lillahi)",
        "حسبنا الله ونعم الوكيل":  "Hasbunallah",
        # --- Common phrases ---
        "ما شاء الله":             "Masha'Allah",
        "ان شاء الله":             "Insha'Allah",
        "بارك الله فيك":           "Barakallahu feek",
        "جزاك الله خيرا":          "Jazakallahu khairan",
        # --- Prophetic titles (observed in lecture output) ---
        "اللهم صل على محمد":      "Salawat (short form)",
        "اللهم صل على سيدنا محمد": "Salawat (formal)",
        "يا رسول الله":            "Ya Rasulallah (address)",
        # --- Famous Hadith openings (REC-4) ---
        "انما الاعمال بالنيات":     "Hadith al-Niyyah (Bukhari #1)",
        "الدين النصيحة":           "Hadith al-Nasihah (Muslim)",
        "من حسن اسلام المرء تركه ما لا يعنيه": "Hadith Husn al-Islam (Tirmidhi)",
        "لا يؤمن احدكم حتى يحب لاخيه": "Hadith: Love for your brother (Bukhari #13)",
        "الدعاء هو العبادة":        "Hadith: Du'a is Worship (Tirmidhi)",
        "الدعاء مخ العبادة":        "Hadith: Du'a is the essence of Worship",
    }.items()
}

# Use the canonical normalised blocklist from arabic_utils.
_NORMALISED_HALLUCINATION_BLOCKLIST = HALLUCINATION_PHRASES_NORMALISED


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_corpus(path: str) -> tuple[tuple[str, ...], tuple[dict, ...], dict[str, int]]:
    """
    Load quran.json and return:
        normalised_verses : tuple[str, ...]  — stripped text used for matching
        metadata          : tuple[dict, ...] — {surah_id, surah_name, ayah_id, text}
        exact_index       : dict[str, int]   — normalised text → index (O(1) lookup)

    Results are cached per unique *path* via @lru_cache.
    Tuples are used instead of lists so the return value is hashable / immutable.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Quran corpus not found at: {path}\n"
            "Download it from https://github.com/risan/quran-json and "
            "place the combined JSON at data/quran.json."
        )

    logger.info("Loading Quran corpus from %s …", path)

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # --- Schema validation (#4 from improvement plan) ---
    if not isinstance(raw, list):
        raise ValueError(
            f"Quran corpus has invalid schema: expected top-level list, "
            f"got {type(raw).__name__}. File may be corrupted."
        )
    if len(raw) != 114:
        logger.warning(
            "Corpus has %d surahs (expected 114) — may be incomplete.",
            len(raw),
        )

    normalised_verses: list[str] = []
    metadata: list[dict] = []

    for surah in raw:
        sid = surah["id"]
        sname = surah.get("name", f"Surah {sid}")
        for verse in surah.get("verses", []):
            aid = verse["id"]
            text = verse.get("text", "")
            normalised_verses.append(normalise_arabic(text))
            metadata.append(
                {
                    "surah_id": sid,
                    "surah_name": sname,
                    "ayah_id": aid,
                    "text": text,
                }
            )

    # Build O(1) exact-lookup index  (CRIT-4 fix)
    exact_index: dict[str, int] = {}
    for idx, norm in enumerate(normalised_verses):
        exact_index.setdefault(norm, idx)   # keep first occurrence

    norm_tuple = tuple(normalised_verses)
    meta_tuple = tuple(metadata)
    logger.info("Corpus loaded — %d verses.", len(norm_tuple))
    return norm_tuple, meta_tuple, exact_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class QuranMatcher:
    """
    Match a transcribed Arabic segment against the Quran corpus.

    Usage
    -----
    matcher = QuranMatcher()                         # uses default corpus path
    result  = matcher.match("بسم الله الرحمن الرحيم")
    if result:
        print(result.surah_name, result.ayah_id, result.confidence)
    """

    def __init__(self, corpus_path: str = _DEFAULT_CORPUS_PATH) -> None:
        self.corpus_path = corpus_path
        # Corpus data lives on the instance (CRIT-3 fix).
        # _load_corpus is @lru_cache'd so multiple instances with the same
        # path share the underlying data without reloading.
        self._normalised, self._metadata, self._exact_index = _load_corpus(corpus_path)

    def match(self, text: str) -> Optional[FormulaMatch | QuranMatch]:
        """
        Attempt to match *text* against known Islamic formulas first, then
        against the Quran corpus.

        Returns
        -------
        FormulaMatch  — if the text is a recognised Islamic formula.
        QuranMatch    — if a corpus match is found (exact or fuzzy).
        None          — if nothing matches.
        """
        if not text or not text.strip():
            return None

        # --- Hallucination filter (REC-3) -----------------------------------
        normalised_check = normalise_arabic(text)
        if normalised_check in _NORMALISED_HALLUCINATION_BLOCKLIST:
            return None

        query = normalised_check

        # --- 0. Islamic formula check (cheap, before corpus) ---------------
        formula = self._match_formula(query, text)
        if formula is not None:
            return formula

        # --- 1. Exact match — O(1) dict lookup (CRIT-4 fix) ----------------
        idx = self._exact_index.get(query)
        if idx is not None and len(query) >= MIN_EXACT_MATCH_CHARS:
            meta = self._metadata[idx]
            return QuranMatch(
                surah_id=meta["surah_id"],
                surah_name=meta["surah_name"],
                ayah_id=meta["ayah_id"],
                original_text=meta["text"],
                matched_text=text,
                confidence=1.0,
                is_exact=True,
            )

        # --- 2. Fuzzy match via rapidfuzz -----------------------------------
        # Skip fuzzy matching for very short queries — they produce too many
        # false positives against short ayahs.  (MED-3 + REC-2 fix)
        if len(query.split()) < 4 or len(query) < MIN_FUZZY_QUERY_CHARS:
            return None

        # Use token_set_ratio for order-independent matching that handles
        # Whisper reorderings and partial ayah transcriptions better than
        # partial_ratio (which falsely matches short strings inside long ones).
        result = rf_process.extractOne(
            query,
            self._normalised,
            scorer=fuzz.token_set_ratio,
            score_cutoff=int(FUZZY_THRESHOLD * 100),
        )

        if result is None:
            return None

        matched_normalised, score, idx = result
        confidence = score / 100.0
        meta = self._metadata[idx]

        return QuranMatch(
            surah_id=meta["surah_id"],
            surah_name=meta["surah_name"],
            ayah_id=meta["ayah_id"],
            original_text=meta["text"],
            matched_text=text,
            confidence=confidence,
            is_exact=False,
        )

    # ------------------------------------------------------------------
    # Formula detection
    # ------------------------------------------------------------------

    @staticmethod
    def _match_formula(normalised_query: str, raw_text: str) -> Optional[FormulaMatch]:
        """
        Check whether *normalised_query* matches (or contains) a known
        Islamic formula.  Uses substring matching so that phrases like
        "قال صلى الله عليه وسلم" still trigger a Salawat detection.

        Returns a FormulaMatch on hit, None otherwise.
        """
        for formula_norm, label in KNOWN_ISLAMIC_FORMULAS.items():
            if formula_norm in normalised_query:
                return FormulaMatch(
                    label=label,
                    matched_text=raw_text,
                    confidence=1.0,
                )
        return None
