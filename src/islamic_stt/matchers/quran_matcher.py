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

from rapidfuzz import fuzz

from islamic_stt.core.arabic_utils import (
    HALLUCINATION_PHRASES_NORMALISED,
    canonicalise_for_matching,
    normalise_arabic,
)

logger = logging.getLogger(__name__)


__all__ = ["QuranMatcher", "QuranMatch", "FormulaMatch"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD = 0.88
MIN_FUZZY_QUERY_CHARS = 20
MIN_EXACT_MATCH_CHARS = 12
_TRIGRAM_CANDIDATE_LIMIT = 80  # max candidates passed to rapidfuzz (Fix 2)
_MIN_FUZZY_COVERAGE = 0.22  # reject tiny fragments against long ayahs
_AMBIGUOUS_FUZZY_DELTA = 1.5  # score points within best treated as ambiguous

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


def _canonicalise(text: str) -> str:
    """Module-level shim for cross-script canonicalization."""
    return canonicalise_for_matching(text)


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
    is_ambiguous: bool = False  # True if multiple ayahs share this text
    ambiguous_count: int = 1  # how many ayahs share this normalized text
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
    canonicalise_for_matching(k): v
    for k, v in {
        # Honorific formulas — safe to recognize without Hadith verification
        "صلى الله عليه وسلم": "Salawat (Durood Ibrahim)",
        "رضي الله عنه": "Radhi Allahu anhu",
        "رضي الله عنها": "Radhi Allahu anha",
        "رضي الله عنهم": "Radhi Allahu anhum",
        "رحمه الله": "Rahimahullah",
        "رحمها الله": "Rahimahallah",
        "جل جلاله": "Jalla Jalaluhu",
        "سبحانه وتعالى": "Subhanahu wa Ta'ala",
        "عز وجل": "Azza wa Jall",
        "تبارك وتعالى": "Tabaraka wa Ta'ala",
        # Dhikr formulas — universally known, no attribution needed
        "الله اكبر": "Takbir",
        "سبحان الله": "Tasbih",
        "الحمد لله": "Tahmid",
        "لا اله الا الله": "Tahlil",
        "استغفر الله": "Istighfar",
        "لا حول ولا قوة الا بالله": "Hawqala",
        "انا لله وانا اليه راجعون": "Istirja (Inna lillahi)",
        "حسبنا الله ونعم الوكيل": "Hasbunallah",
        "ما شاء الله": "Masha'Allah",
        "ان شاء الله": "Insha'Allah",
        "بارك الله فيك": "Barakallahu feek",
        "جزاك الله خيرا": "Jazakallahu khairan",
        # Salawat forms
        "اللهم صل على محمد": "Salawat (short form)",
        "اللهم صل على سيدنا محمد": "Salawat (formal)",
        "يا رسول الله": "Ya Rasulallah (address)",
        # Urdu-script honorifics (Whisper lecture output)
        "اللہم صلی اللہ علیہ وسلم": "Salawat (Urdu script)",
        "صلی اللہ علیہ وسلم": "Salawat (Urdu script)",
        "اللہم صل علیہ وسلم": "Salawat (Urdu script)",
        "رضی اللہ عنہ": "Radhi Allahu anhu (Urdu script)",
        "رضی اللہ عنها": "Radhi Allahu anha (Urdu script)",
        "رضی اللہ عنهم": "Radhi Allahu anhum (Urdu script)",
        "بسم اللہ الرحمن الرحیم": "Basmala (Urdu script)",
        "الحمدللہ": "Tahmid (merged)",
        "سبحان اللہ": "Tasbih (Urdu script)",
        # P0 FIX: Hadith texts REMOVED from here.
        # They must go through the Hadith DB verifier to get proper
        # collection/number citations and paraphrase detection.
        # Previously: "انما الاعمال بالنيات", "الدين النصيحة", etc.
        # were attributed as formulas with 100% confidence, bypassing
        # Hadith verification entirely — a false attribution risk.
    }.items()
}

# Build a canonicalised version of the hallucination blocklist so that
# Urdu-orthography hallucinations are also caught.
_NORMALISED_HALLUCINATION_BLOCKLIST = HALLUCINATION_PHRASES_NORMALISED | frozenset(
    canonicalise_for_matching(p) for p in HALLUCINATION_PHRASES_NORMALISED
)


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


def _adaptive_fuzzy_threshold(avg_word_confidence: float | None) -> float:
    """Lower fuzzy bar slightly when ASR word confidence is moderate."""
    if avg_word_confidence is None:
        return FUZZY_THRESHOLD
    if avg_word_confidence >= 0.75:
        return FUZZY_THRESHOLD
    if avg_word_confidence >= 0.55:
        return 0.82
    return FUZZY_THRESHOLD


def _sliding_word_windows(query: str, *, min_words: int = 5, max_words: int = 12) -> list[str]:
    """Rolling word windows for partial-ayah matching inside long segments."""
    words = query.split()
    if len(words) < min_words:
        return []
    windows: list[str] = []
    upper = min(max_words, len(words))
    for size in range(upper, min_words - 1, -1):
        for i in range(len(words) - size + 1):
            window = " ".join(words[i : i + size])
            if len(window) >= MIN_FUZZY_QUERY_CHARS:
                windows.append(window)
    return windows


# ---------------------------------------------------------------------------
# Trigram helpers (Fix 2)
# ---------------------------------------------------------------------------


def _trigrams(text: str) -> list[str]:
    """Return all character 3-grams of *text* (no padding)."""
    return [text[i : i + 3] for i in range(len(text) - 2)]


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
) -> tuple[tuple[str, ...], tuple[dict, ...], dict[str, list[int]], dict[str, set[int]]]:
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
            normalised_verses.append(canonicalise_for_matching(text))
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
        self._normalised, self._metadata, self._exact_index, self._trigram_index = _load_corpus(
            corpus_path
        )

    def match(
        self,
        text: str,
        *,
        normalised: str | None = None,
        canonicalised: str | None = None,
        allow_fuzzy: bool = True,
        avg_word_confidence: float | None = None,
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

        # Use cross-script canonicalization for all matching.
        # This maps Urdu-script chars (ہ→ه, ی→ي, ک→ك) to Arabic
        # equivalents so Whisper's Urdu-orthography output matches
        # the Arabic Quran corpus.
        query = canonicalised if canonicalised is not None else canonicalise_for_matching(text)

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
                    alt_refs.append(
                        {
                            "surah_id": alt_meta["surah_id"],
                            "surah_name": alt_meta["surah_name"],
                            "ayah_id": alt_meta["ayah_id"],
                        }
                    )

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

        # --- 2. Fuzzy match (full query, then sliding windows) --------------
        if allow_fuzzy:
            fuzzy = self._fuzzy_match_query(
                query,
                text,
                avg_word_confidence=avg_word_confidence,
            )
            if fuzzy is not None:
                return fuzzy

            if len(query.split()) >= 6:
                best_window: QuranMatch | None = None
                for window in _sliding_word_windows(query):
                    if window == query:
                        continue
                    hit = self._fuzzy_match_query(
                        window,
                        text,
                        avg_word_confidence=avg_word_confidence,
                    )
                    if hit is not None and isinstance(hit, QuranMatch):
                        if best_window is None or hit.confidence > best_window.confidence:
                            best_window = hit
                if best_window is not None:
                    return best_window

        return None

    def _fuzzy_match_query(
        self,
        query: str,
        raw_text: str,
        *,
        avg_word_confidence: float | None,
    ) -> QuranMatch | None:
        """Fuzzy-match a single normalised query string against the corpus."""
        if len(query.split()) < 4 or len(query) < MIN_FUZZY_QUERY_CHARS:
            return None

        candidates = _trigram_candidates(query, self._trigram_index, k=_TRIGRAM_CANDIDATE_LIMIT)
        candidate_indices = list(range(len(self._normalised))) if not candidates else candidates

        scored_matches: list[tuple[float, int, float]] = []
        threshold = _adaptive_fuzzy_threshold(avg_word_confidence) * 100

        for ci in candidate_indices:
            corpus_text = self._normalised[ci]
            coverage = min(len(query), len(corpus_text)) / max(len(query), len(corpus_text))
            if coverage < _MIN_FUZZY_COVERAGE:
                continue

            ts_score = fuzz.token_set_ratio(query, corpus_text)
            if ts_score < threshold * 0.85:
                continue

            lev_ratio = fuzz.ratio(query, corpus_text)
            partial_score = fuzz.partial_ratio(query, corpus_text)
            adjusted = 0.35 * ts_score + 0.45 * lev_ratio + 0.20 * partial_score

            if len(query.split()) >= 5 and (query in corpus_text or corpus_text in query):
                adjusted = max(adjusted, 90.0 + (8.0 * coverage))

            if adjusted >= threshold:
                scored_matches.append((adjusted, ci, coverage))

        if not scored_matches:
            return None

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        best_adjusted, best_idx, best_coverage = scored_matches[0]
        close_matches = [
            (score, idx, coverage)
            for score, idx, coverage in scored_matches
            if best_adjusted - score <= _AMBIGUOUS_FUZZY_DELTA
        ]

        confidence = (best_adjusted / 100.0) * (0.75 + 0.25 * best_coverage)
        meta = self._metadata[best_idx]
        is_ambiguous = len(close_matches) > 1
        alt_refs = None
        if is_ambiguous:
            alt_refs = []
            seen: set[tuple[int, int]] = set()
            for _, alt_idx, _ in close_matches[:12]:
                alt_meta = self._metadata[alt_idx]
                key = (alt_meta["surah_id"], alt_meta["ayah_id"])
                if key in seen:
                    continue
                seen.add(key)
                alt_refs.append(
                    {
                        "surah_id": alt_meta["surah_id"],
                        "surah_name": alt_meta["surah_name"],
                        "ayah_id": alt_meta["ayah_id"],
                    }
                )

        return QuranMatch(
            surah_id=meta["surah_id"],
            surah_name=meta["surah_name"],
            ayah_id=meta["ayah_id"],
            original_text=meta["text"],
            matched_text=raw_text,
            confidence=round(min(confidence, 0.98), 4),
            is_exact=False,
            is_ambiguous=is_ambiguous,
            ambiguous_count=len(close_matches),
            alternate_refs=alt_refs,
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
