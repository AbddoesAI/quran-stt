"""
hadith_matcher.py
-----------------
Fuzzy-matches a transcribed Arabic segment against the Sunnah.com API.

Sunnah.com API
--------------
Base URL  : https://api.sunnah.com/v1
Auth      : requires a free API key (header: X-API-Key)
Endpoint  : GET /hadiths/search?q=<text>&limit=<n>

The API returns hadiths in both Arabic and English.  We match against the
Arabic `body` field.

Matching strategy
-----------------
1.  Send the normalised query to the API (rate-limited with retry).
2.  For each returned hadith, compute rapidfuzz.fuzz.partial_ratio between
    the normalised query and the normalised Arabic body.
3.  Return the best match above FUZZY_THRESHOLD, or None.

Rate limiting
-------------
The free tier allows 100 requests / hour.  We add a small sleep between
calls and honour 429 responses with exponential backoff.
"""

from __future__ import annotations

import logging
import os
import time
import shelve
from dataclasses import dataclass
from typing import Optional

import requests
from rapidfuzz import fuzz

from islamic_stt.core.arabic_utils import normalise_arabic

logger = logging.getLogger(__name__)


__all__ = ["HadithMatcher", "HadithMatch"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUNNAH_API_BASE = "https://api.sunnah.com/v1"
SUNNAH_API_KEY_ENV = "SUNNAH_API_KEY"    # set this in your shell / Colab secrets

# Minimum rapidfuzz partial_ratio score (0–100) to accept a hadith match.
FUZZY_THRESHOLD = 78   # slightly lower than Quran threshold; hadith transcription
                       # is noisier (chain text, narrator names, etc.)

# How many API results to retrieve per query.
_SEARCH_LIMIT = 5

# Seconds to wait between API requests (politeness / rate-limit headroom).
_REQUEST_DELAY = 1.0

# Retry settings for 429 / transient errors.
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0     # seconds; doubles on each retry
_CACHE_PATH_ENV = "HADITH_CACHE_PATH"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HadithMatch:
    collection: str           # e.g. "bukhari", "muslim"
    hadith_number: str        # as returned by the API
    arabic_text: str          # original Arabic body from API
    english_text: str         # English translation (useful for verification)
    matched_text: str         # what the transcription contained
    confidence: float         # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HadithMatcher:
    """
    Query the Sunnah.com API and fuzzy-match a transcribed Arabic segment.

    Parameters
    ----------
    api_key : Sunnah.com API key.  If None, reads from the environment
              variable SUNNAH_API_KEY.  Raises ValueError if not found.

    Usage
    -----
    matcher = HadithMatcher()
    result  = matcher.match("إنما الأعمال بالنيات")
    if result:
        print(result.collection, result.hadith_number, result.confidence)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get(SUNNAH_API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"Sunnah.com API key not provided.  Set the environment "
                f"variable {SUNNAH_API_KEY_ENV} or pass api_key= explicitly.\n"
                "Get a free key at https://sunnah.com/developers"
            )

        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-API-Key": self.api_key,
                "Accept": "application/json",
            }
        )
        self._last_request_time: float = 0.0
        self._cache_path = os.environ.get(_CACHE_PATH_ENV, os.path.join("cache", "hadith_cache.db"))
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "HadithMatcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def match(self, text: str) -> Optional[HadithMatch]:
        """
        Search for *text* in the Sunnah.com hadith database.

        Returns the best HadithMatch above the confidence threshold, or None.
        """
        if not text or not text.strip():
            return None

        query_normalised = normalise_arabic(text)

        candidates = self._cached_search(query_normalised)
        if not candidates:
            return None

        best_match: Optional[HadithMatch] = None
        best_score = 0

        for hadith in candidates:
            arabic_body = hadith.get("arabic", {}).get("body", "")
            if not arabic_body:
                continue

            corpus_normalised = normalise_arabic(arabic_body)
            score = fuzz.partial_ratio(query_normalised, corpus_normalised)

            if score > best_score:
                best_score = score
                english_body = hadith.get("english", {}).get("body", "")
                best_match = HadithMatch(
                    collection=hadith.get("collection", "unknown"),
                    hadith_number=str(hadith.get("hadithNumber", "?")),
                    arabic_text=arabic_body,
                    english_text=english_body,
                    matched_text=text,
                    confidence=score / 100.0,
                )

        if best_match and best_score >= FUZZY_THRESHOLD:
            return best_match

        return None

    def _cached_search(self, query: str) -> list[dict]:
        if len(query) < 12:
            return []
        try:
            with shelve.open(self._cache_path) as db:
                cached = db.get(query)
                if cached is not None:
                    return cached
                result = self._search(query)
                db[query] = result
                return result
        except Exception as exc:
            logger.warning("Cache error (%s), falling back to API.", exc)
            return self._search(query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search(self, query: str) -> list[dict]:
        """
        Hit the Sunnah.com search endpoint with retry / backoff.
        Returns a list of hadith dicts, or [] on failure.
        """
        # Throttle: ensure minimum gap between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)

        url = f"{SUNNAH_API_BASE}/hadiths/search"
        params = {"q": query, "limit": _SEARCH_LIMIT}

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._session.get(url, params=params, timeout=10)
                self._last_request_time = time.time()

                if response.status_code == 200:
                    data = response.json()
                    # The API wraps results under 'data'
                    return data.get("data", [])

                if response.status_code == 429:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Rate limited. Waiting %ds (attempt %d/%d) …",
                        int(wait), attempt, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                # Any other HTTP error: log and give up
                logger.error(
                    "API error %d: %s",
                    response.status_code, response.text[:200],
                )
                return []

            except requests.RequestException as exc:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Network error (%s). Retrying in %ds …",
                    exc, int(wait),
                )
                time.sleep(wait)

        logger.error("All retries exhausted. Returning empty result.")
        return []
