"""
hadith_db.py
------------
Local SQLite + FTS5 Hadith matcher with hybrid retrieval.

Replaces the Sunnah.com API as the primary Hadith matching engine.
The API is kept as an optional fallback only.

Matching strategy (HADITH-2 fix: two-stage scorer)
---------------------------------------------------
1. FTS5 full-text search on normalized Arabic → top-20 candidates
2. Trigram overlap pre-filter → top-10 candidates
3. Two-stage fuzzy scoring:
   a. fuzz.partial_ratio as fast filter (threshold 70)
   b. fuzz.token_set_ratio as precision scorer (threshold 82)
4. Length ratio check — reject if query < 15% of candidate
5. Confidence calibration (HADITH-3)
6. Paraphrase detection (HADITH-5)

Performance
-----------
All matching is local (no network). Typical query: <10ms on SSD.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from islamic_stt.core.arabic_utils import normalise_arabic

logger = logging.getLogger(__name__)

__all__ = ["LocalHadithMatcher", "HadithMatch"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum normalized query length to attempt matching
_MIN_QUERY_CHARS = 15

# FTS5 result limit (broad initial retrieval)
_FTS_LIMIT = 20

# Two-stage fuzzy thresholds
_PARTIAL_RATIO_THRESHOLD = 70    # fast filter
_TOKEN_SET_RATIO_THRESHOLD = 82  # precision scorer

# Length ratio: reject if query is less than this fraction of corpus text
_MIN_LENGTH_RATIO = 0.10

# Paraphrase detection thresholds
_PARAPHRASE_CONFIDENCE = 0.90
_PARAPHRASE_LENGTH_DIFF = 0.30  # 30% length difference triggers paraphrase flag

_DEFAULT_DB_PATH = os.path.join("data", "hadith.db")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HadithMatch:
    """Result of a Hadith match."""
    collection: str           # e.g. "bukhari", "muslim"
    hadith_number: str        # as stored in the DB
    arabic_text: str          # original Arabic body
    english_text: str         # English translation
    matched_text: str         # what the transcription contained
    confidence: float         # 0.0 – 1.0 (calibrated)
    is_paraphrase: bool       # True if likely a paraphrase, not exact quote
    chapter: str = ""         # chapter name/number


# ---------------------------------------------------------------------------
# Trigram helpers for candidate re-ranking
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> list[str]:
    """Return character 3-grams."""
    return [text[i:i + 3] for i in range(len(text) - 2)]


def _trigram_overlap(query: str, candidate: str) -> float:
    """Jaccard similarity of trigram sets."""
    q_tg = set(_trigrams(query))
    c_tg = set(_trigrams(candidate))
    if not q_tg or not c_tg:
        return 0.0
    intersection = len(q_tg & c_tg)
    union = len(q_tg | c_tg)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Confidence calibration (HADITH-3)
# ---------------------------------------------------------------------------

def _calibrate_confidence(
    fuzzy_score: float,
    query_len: int,
    corpus_len: int,
) -> float:
    """
    Calibrate raw fuzzy score into a meaningful confidence value.

    Raw fuzzy scores are not probabilities. A 78% partial_ratio against
    a very long hadith text is less meaningful than 78% against a
    similar-length text.

    Adjustment factors:
    - Length ratio: penalizes when query is much shorter than corpus text
    - Score scaling: maps [threshold..100] → [0.5..1.0]
    """
    if corpus_len == 0:
        return 0.0

    # Length ratio penalty
    length_ratio = min(query_len, corpus_len) / max(query_len, corpus_len)

    # Combine: weight fuzzy score at 70%, length ratio at 30%
    adjusted = (fuzzy_score / 100.0) * (0.70 + 0.30 * length_ratio)

    return round(min(1.0, max(0.0, adjusted)), 4)


# ---------------------------------------------------------------------------
# Paraphrase detection (HADITH-5)
# ---------------------------------------------------------------------------

def _is_paraphrase(
    confidence: float,
    query_len: int,
    corpus_len: int,
    token_set_score: float,
    token_sort_score: float,
) -> bool:
    """
    Detect if a match is likely a paraphrase rather than a direct quote.

    Indicators of paraphrase:
    1. Confidence below threshold
    2. Significant length difference
    3. High token_set but low token_sort (words present but order differs)
    """
    if confidence >= _PARAPHRASE_CONFIDENCE:
        return False

    length_diff = abs(query_len - corpus_len) / max(query_len, corpus_len)
    if length_diff > _PARAPHRASE_LENGTH_DIFF:
        return True

    # Word order difference: token_set ignores order, token_sort respects it
    order_delta = token_set_score - token_sort_score
    if order_delta > 15:  # >15 point difference suggests reordered words
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LocalHadithMatcher:
    """
    Match transcribed Arabic text against a local SQLite Hadith database.

    Usage
    -----
    matcher = LocalHadithMatcher()
    result = matcher.match("إنما الأعمال بالنيات")
    if result:
        print(result.collection, result.hadith_number, result.confidence)
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self.db_path = db_path

        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"Hadith database not found at: {db_path}\n"
                "Build it with: python scripts/build_hadith_db.py"
            )

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA cache_size=-32000")

        count = self._conn.execute("SELECT COUNT(*) FROM hadiths").fetchone()[0]
        logger.info("Local Hadith DB loaded: %d hadiths from %s", count, db_path)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> LocalHadithMatcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Main matching method
    # ------------------------------------------------------------------

    def match(self, text: str) -> HadithMatch | None:
        """
        Search for *text* in the local Hadith database.

        Returns the best HadithMatch above the confidence threshold, or None.
        """
        if not text or not text.strip():
            return None

        query_normalized = normalise_arabic(text)
        if len(query_normalized) < _MIN_QUERY_CHARS:
            return None

        # Stage 1: FTS5 full-text search for broad candidates
        candidates = self._fts_search(query_normalized)

        if not candidates:
            words = query_normalized.split()
            if len(words) >= 3:
                partial_query = " ".join(words[:3])
                candidates = self._fts_search(partial_query, limit=_FTS_LIMIT * 2)

        if not candidates:
            return None

        # P1 fix: Rerank FTS candidates by trigram overlap before fuzzy scoring.
        # This was documented but never actually used.
        if len(candidates) > 5:
            scored = [
                (c, _trigram_overlap(query_normalized, c["text_ar_normalized"]))
                for c in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [c for c, _ in scored[:10]]  # top 10 by trigram

        # Stage 2: Two-stage fuzzy scoring
        return self._score_candidates(candidates, text, query_normalized)

    def match_many(self, texts: list[str]) -> list[HadithMatch | None]:
        """Match multiple texts (all local, no network needed)."""
        return [self.match(t) for t in texts]

    # ------------------------------------------------------------------
    # FTS5 search
    # ------------------------------------------------------------------

    def _fts_search(self, query: str, limit: int = _FTS_LIMIT) -> list[dict]:
        """
        Use FTS5 to find candidate hadiths matching the query.
        """
        try:
            # FTS5 MATCH query — use individual words joined with OR
            words = query.split()
            if not words:
                return []

            # Use the most significant words (longest ones)
            significant = sorted(words, key=len, reverse=True)[:6]
            fts_query = " OR ".join(f'"{w}"' for w in significant if len(w) >= 3)

            if not fts_query:
                return []

            rows = self._conn.execute(
                """
                SELECT h.id, h.collection, h.hadith_no, h.chapter,
                       h.text_ar, h.text_ar_normalized, h.text_en
                FROM hadith_fts f
                JOIN hadiths h ON h.id = f.rowid
                WHERE hadith_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

            return [dict(r) for r in rows]

        except sqlite3.OperationalError as exc:
            logger.warning("FTS search error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Two-stage fuzzy scoring (HADITH-2 fix)
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: list[dict],
        raw_text: str,
        query_normalized: str,
    ) -> HadithMatch | None:
        """
        Two-stage fuzzy scoring:
        1. partial_ratio as fast filter
        2. token_set_ratio as precision scorer
        + length ratio check
        + confidence calibration
        + paraphrase detection
        """
        query_len = len(query_normalized)
        best_match: HadithMatch | None = None
        best_calibrated = 0.0

        for candidate in candidates:
            corpus_normalized = candidate["text_ar_normalized"]
            corpus_len = len(corpus_normalized)

            # Length ratio check: skip if query is < 10% of corpus
            if corpus_len > 0:
                ratio = query_len / corpus_len
                if ratio < _MIN_LENGTH_RATIO and query_len < corpus_len:
                    continue

            # Stage 1: fast filter with partial_ratio
            partial_score = fuzz.partial_ratio(query_normalized, corpus_normalized)
            if partial_score < _PARTIAL_RATIO_THRESHOLD:
                continue

            # Stage 2: precision with token_set_ratio
            token_set_score = fuzz.token_set_ratio(query_normalized, corpus_normalized)
            if token_set_score < _TOKEN_SET_RATIO_THRESHOLD:
                continue

            # Also compute token_sort for paraphrase detection
            token_sort_score = fuzz.token_sort_ratio(query_normalized, corpus_normalized)

            # Confidence calibration (HADITH-3)
            calibrated = _calibrate_confidence(
                max(partial_score, token_set_score),
                query_len,
                corpus_len,
            )

            if calibrated > best_calibrated:
                best_calibrated = calibrated

                # Paraphrase detection (HADITH-5)
                paraphrase = _is_paraphrase(
                    calibrated,
                    query_len,
                    corpus_len,
                    token_set_score,
                    token_sort_score,
                )

                best_match = HadithMatch(
                    collection=candidate["collection"],
                    hadith_number=str(candidate["hadith_no"]),
                    arabic_text=candidate["text_ar"],
                    english_text=candidate.get("text_en", ""),
                    matched_text=raw_text,
                    confidence=calibrated,
                    is_paraphrase=paraphrase,
                    chapter=candidate.get("chapter", ""),
                )

        return best_match
