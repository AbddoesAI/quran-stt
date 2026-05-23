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
3.  Trigram candidate filtering → narrows ~6236 verses to 20–100 candidates.
4.  Rapid fuzzy match using rapidfuzz.fuzz.token_set_ratio on candidates only.
5.  Returns None when the best score is below FUZZY_THRESHOLD.

Changes (audit fixes)
---------------------
- Fix 2  : Trigram inverted index built at corpus load; fuzzy match now scans
           ~20–100 candidates instead of all 6236 verses (10–50x speedup).
- Fix 3  : `match()` accepts pre-normalised text via `normalised` kwarg to
           avoid redundant regex pipelines when called from pipeline.py.
- Fix 6  : Aho-Corasick automaton for O(n) multi-pattern formula scan
           (falls back to linear scan if pyahocorasick unavailable).
- Fix 10 : Modernised type annotations to Python 3.10+ style.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from rapidfuzz import fuzz, process as rf_process
from islamic_stt.core.arabic_utils import (
    normalise_arabic,
    HALLUCINATION_PHRASES_RAW,
    HALLUCINATION_PHRASES_NORMALISED,
)

logger = logging.getLogger(__name__)


__all__ = ["QuranMatcher", "QuranMatch", "FormulaMatch"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD = 0.88
MIN_FUZZY_QUERY_CHARS = 20
MIN_EXACT_MATCH_CHARS = 12
_TRIGRAM_CANDIDATE_LIMIT = 80    # max candidates passed to rapidfuzz (Fix 2)

_DEFAULT_CORPUS_PATH = os.path.join("data", "quran.json")


# ---------------------------------------------------------------------------
# Optional Aho-Corasick (Fix 6)
# ---------------------------------------------------------------------------

try:
    import ahocorasick as _ac
    _AC_AVAILABLE = True
except ImportError:
    _AC_AVAILABLE = False
    _ac = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Normalisation shim (kept for any external code that imported this)
# ---------------------------------------------------------------------------

def _normalise_arabic(text: str) -> str:
    """Backward-compatible shim."""
    return normalise_arabic(text)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QuranMatch:
    surah_id: int
    surah_name: str
    ayah_id: int
    original_text: str
    matched_text: str
    confidence: float
    is_exact: bool
    is_ambiguous: bool = False       # True if multiple ayahs share this text
    ambiguous_count: int = 1         # how many ayahs share this normalized text
    alternate_refs: list | None = None  # list of {surah_id, surah_name, ayah_id} dicts


@dataclass
class FormulaMatch:
    """A recognised Islamic formula (Salawat, honorific, dhikr, etc.)."""
    label: str
    matched_text: str
    confidence: float


# ---------------------------------------------------------------------------
# Known Islamic Formulas
# ---------------------------------------------------------------------------

KNOWN_ISLAMIC_FORMULAS: dict[str, str] = {
    normalise_arabic(k): v
    for k, v in {
        # Honorific formulas — safe to recognize without Hadith verification
        "صلى الله عليه وسلم":    "Salawat (Durood Ibrahim)",
        "رضي الله عنه":          "Radhi Allahu anhu",
        "رضي الله عنها":         "Radhi Allahu anha",
        "رضي الله عنهم":         "Radhi Allahu anhum",
        "رحمه الله":             "Rahimahullah",
        "رحمها الله":            "Rahimahallah",
        "جل جلاله":              "Jalla Jalaluhu",
        "سبحانه وتعالى":         "Subhanahu wa Ta'ala",
        "عز وجل":                "Azza wa Jall",
        "تبارك وتعالى":          "Tabaraka wa Ta'ala",
        # Dhikr formulas — universally known, no attribution needed
        "الله اكبر":              "Takbir",
        "سبحان الله":             "Tasbih",
        "الحمد لله":              "Tahmid",
        "لا اله الا الله":         "Tahlil",
        "استغفر الله":            "Istighfar",
        "لا حول ولا قوة الا بالله": "Hawqala",
        "انا لله وانا اليه راجعون": "Istirja (Inna lillahi)",
        "حسبنا الله ونعم الوكيل":  "Hasbunallah",
        "ما شاء الله":             "Masha'Allah",
        "ان شاء الله":             "Insha'Allah",
        "بارك الله فيك":           "Barakallahu feek",
        "جزاك الله خيرا":          "Jazakallahu khairan",
        # Salawat forms
        "اللهم صل على محمد":      "Salawat (short form)",
        "اللهم صل على سيدنا محمد": "Salawat (formal)",
        "يا رسول الله":            "Ya Rasulallah (address)",
        # P0 FIX: Hadith texts REMOVED from here.
        # They must go through the Hadith DB verifier to get proper
        # collection/number citations and paraphrase detection.
        # Previously: "انما الاعمال بالنيات", "الدين النصيحة", etc.
        # were attributed as formulas with 100% confidence, bypassing
        # Hadith verification entirely — a false attribution risk.
    }.items()
}

_NORMALISED_HALLUCINATION_BLOCKLIST = HALLUCINATION_PHRASES_NORMALISED


# ---------------------------------------------------------------------------
# Aho-Corasick automaton (Fix 6) — built once at module load
# ---------------------------------------------------------------------------

def _build_formula_automaton() -> object | None:
    """Build an Aho-Corasick automaton over known formula keys."""
    if not _AC_AVAILABLE:
        return None
    automaton = _ac.Automaton()
    for norm_key, label in KNOWN_ISLAMIC_FORMULAS.items():
        automaton.add_word(norm_key, (norm_key, label))
    automaton.make_automaton()
    return automaton


_FORMULA_AUTOMATON = _build_formula_automaton()


# ---------------------------------------------------------------------------
# Trigram helpers (Fix 2)
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> list[str]:
    """Return all character 3-grams of *text* (no padding)."""
    return [text[i:i + 3] for i in range(len(text) - 2)]


def _build_trigram_index(verses: tuple[str, ...]) -> dict[str, set[int]]:
    """
    Build an inverted index: trigram → set of verse indices.

    Built once at corpus load time and cached with the corpus data.
    """
    index: dict[str, set[int]] = {}
    for idx, verse in enumerate(verses):
        for tg in _trigrams(verse):
            index.setdefault(tg, set()).add(idx)
    return index


def _trigram_candidates(
    query: str,
    index: dict[str, set[int]],
    k: int = _TRIGRAM_CANDIDATE_LIMIT,
) -> list[int]:
    """
    Return up to *k* verse indices most likely to match *query*.

    Uses trigram overlap scoring (Counter) to rank candidates.
    Returns an empty list if query is too short for trigrams.
    """
    if len(query) < 3:
        return []
    scores: Counter = Counter()
    for tg in _trigrams(query):
        scores.update(index.get(tg, set()))
    return [idx for idx, _ in scores.most_common(k)]


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_corpus(
    path: str,
) -> tuple[tuple[str, ...], tuple[dict, ...], dict[str, int], dict[str, set[int]]]:
    """
    Load quran.json and return:
        normalised_verses : tuple[str, ...]
        metadata          : tuple[dict, ...]
        exact_index       : dict[str, int]         — O(1) exact lookup
        trigram_index     : dict[str, set[int]]    — Fix 2 candidate filter

    Results are cached per unique *path* via @lru_cache.
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

    if not isinstance(raw, list):
        raise ValueError(
            f"Quran corpus has invalid schema: expected top-level list, "
            f"got {type(raw).__name__}. File may be corrupted."
        )
    if len(raw) != 114:
        logger.warning("Corpus has %d surahs (expected 114) — may be incomplete.", len(raw))

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

    # P0 fix: store list[int] so repeated ayahs return all candidates
    exact_index: dict[str, list[int]] = {}
    for idx, norm in enumerate(normalised_verses):
        exact_index.setdefault(norm, []).append(idx)

    norm_tuple = tuple(normalised_verses)
    meta_tuple = tuple(metadata)

    # Build trigram index (Fix 2)
    trigram_index = _build_trigram_index(norm_tuple)

    logger.info("Corpus loaded — %d verses, trigram index built.", len(norm_tuple))
    return norm_tuple, meta_tuple, exact_index, trigram_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class QuranMatcher:
    """
    Match a transcribed Arabic segment against the Quran corpus.

    Usage
    -----
    matcher = QuranMatcher()
    result  = matcher.match("بسم الله الرحمن الرحيم")
    if result:
        print(result.surah_name, result.ayah_id, result.confidence)
    """

    def __init__(self, corpus_path: str = _DEFAULT_CORPUS_PATH) -> None:
        self.corpus_path = corpus_path
        self._normalised, self._metadata, self._exact_index, self._trigram_index = (
            _load_corpus(corpus_path)
        )

    def match(
        self,
        text: str,
        *,
        normalised: str | None = None,
    ) -> FormulaMatch | QuranMatch | None:
        """
        Attempt to match *text* against known Islamic formulas first, then
        against the Quran corpus.

        Parameters
        ----------
        text       : Raw transcribed segment text.
        normalised : Pre-computed normalised form of *text* (Fix 3).  If
                     provided, skips redundant normalisation call.

        Returns
        -------
        FormulaMatch  — if the text is a recognised Islamic formula.
        QuranMatch    — if a corpus match is found (exact or fuzzy).
        None          — if nothing matches.
        """
        if not text or not text.strip():
            return None

        query = normalised if normalised is not None else normalise_arabic(text)

        # --- Hallucination filter ---
        if query in _NORMALISED_HALLUCINATION_BLOCKLIST:
            return None

        # --- 0. Islamic formula check (Aho-Corasick or linear fallback) -----
        formula = self._match_formula(query, text)
        if formula is not None:
            return formula

        # --- 1. Exact match — O(1) dict lookup ------------------------------
        # P0 review 4: For ambiguous matches, include ALL candidate references
        # and set is_exact=False since we can't be sure which ayah is correct.
        indices = self._exact_index.get(query)
        if indices and len(query) >= MIN_EXACT_MATCH_CHARS:
            idx = indices[0]
            meta = self._metadata[idx]
            is_ambiguous = len(indices) > 1

            # Build alternate refs for ALL candidates
            alt_refs = None
            if is_ambiguous:
                alt_refs = []
                for alt_idx in indices:
                    alt_meta = self._metadata[alt_idx]
                    alt_refs.append({
                        "surah_id": alt_meta["surah_id"],
                        "surah_name": alt_meta["surah_name"],
                        "ayah_id": alt_meta["ayah_id"],
                    })

            return QuranMatch(
                surah_id=meta["surah_id"],
                surah_name=meta["surah_name"],
                ayah_id=meta["ayah_id"],
                original_text=meta["text"],
                matched_text=text,
                # Ambiguous: reduce confidence, mark not-exact since citation is uncertain
                confidence=0.90 if is_ambiguous else 1.0,
                is_exact=not is_ambiguous,  # can't be "exact" if we don't know which ayah
                is_ambiguous=is_ambiguous,
                ambiguous_count=len(indices),
                alternate_refs=alt_refs,
            )

        # --- 2. Fuzzy match with trigram pre-filtering (Fix 2) --------------
        if len(query.split()) < 4 or len(query) < MIN_FUZZY_QUERY_CHARS:
            return None

        candidates = _trigram_candidates(query, self._trigram_index, k=_TRIGRAM_CANDIDATE_LIMIT)

        if not candidates:
            logger.debug("Trigram returned no candidates — falling back to full scan.")
            candidate_verses = self._normalised
            result = rf_process.extractOne(
                query,
                candidate_verses,
                scorer=fuzz.token_set_ratio,
                score_cutoff=int(FUZZY_THRESHOLD * 100),
            )
            if result is None:
                return None
            _, score, match_idx = result
        else:
            candidate_verses = [self._normalised[i] for i in candidates]
            result = rf_process.extractOne(
                query,
                candidate_verses,
                scorer=fuzz.token_set_ratio,
                score_cutoff=int(FUZZY_THRESHOLD * 100),
            )
            if result is None:
                return None
            _, score, local_idx = result
            match_idx = candidates[local_idx]

        # P1 review 4: True ordered alignment using normalized Levenshtein.
        # token_sort_ratio sorts tokens (destroys order), so it can't
        # reliably protect Quranic word order. Levenshtein on the raw
        # normalized strings respects insertion/deletion/substitution order.
        corpus_text = self._normalised[match_idx]
        lev_ratio = fuzz.ratio(query, corpus_text)  # char-level Levenshtein ratio

        # Contiguous coverage: does the query appear as a substring?
        # partial_ratio already measures this, but we use it as a
        # separate signal for short sacred phrases.
        partial_score = fuzz.partial_ratio(query, corpus_text)

        # Combine: token_set gives vocabulary overlap, Levenshtein gives
        # ordered similarity, partial_ratio gives substring coverage.
        # Weight: 40% token_set + 40% Levenshtein + 20% partial
        adjusted_score = (
            0.40 * score
            + 0.40 * lev_ratio
            + 0.20 * partial_score
        )

        if adjusted_score < FUZZY_THRESHOLD * 100:
            return None

        confidence = adjusted_score / 100.0
        meta = self._metadata[match_idx]

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
    # Formula detection (Fix 6 — Aho-Corasick)
    # ------------------------------------------------------------------

    @staticmethod
    def _match_formula(normalised_query: str, raw_text: str) -> FormulaMatch | None:
        """
        Check whether *normalised_query* matches (or contains) a known
        Islamic formula.

        Uses Aho-Corasick when pyahocorasick is installed (O(n) scan);
        falls back to linear substring loop otherwise.
        """
        if _FORMULA_AUTOMATON is not None:
            for _, (_, label) in _FORMULA_AUTOMATON.iter(normalised_query):
                return FormulaMatch(label=label, matched_text=raw_text, confidence=1.0)
            return None

        # Linear fallback
        for formula_norm, label in KNOWN_ISLAMIC_FORMULAS.items():
            if formula_norm in normalised_query:
                return FormulaMatch(label=label, matched_text=raw_text, confidence=1.0)
        return None
