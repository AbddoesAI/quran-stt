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

Changes (audit fixes)
---------------------
- Fix 1  : Replaced `shelve` with `diskcache.Cache` (thread-safe, TTL-bounded,
           no corruption under ThreadPoolExecutor concurrent writes).
- Fix 9  : Added `match_many_async()` coroutine using `httpx.AsyncClient` for
           I/O-bound concurrent Hadith lookups.
- Fix 10 : Modernised type annotations to Python 3.10+ style (X | None, list[X]).
- Fix 11 : Cache path sanitized via pathlib.Path.resolve().
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
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

# diskcache settings (Fix 1)
_CACHE_TTL_SECONDS = 60 * 60 * 24 * 30   # 30-day TTL per entry
_CACHE_SIZE_LIMIT = 256 * 1024 * 1024     # 256 MB max on-disk size


# ---------------------------------------------------------------------------
# Optional httpx import (Fix 9)
# ---------------------------------------------------------------------------

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    _httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# diskcache import (Fix 1)
# ---------------------------------------------------------------------------

try:
    import diskcache as _diskcache
    _DISKCACHE_AVAILABLE = True
except ImportError:
    _DISKCACHE_AVAILABLE = False
    _diskcache = None  # type: ignore[assignment]
    logger.warning(
        "diskcache not installed — falling back to in-memory dict cache. "
        "Install with: pip install diskcache"
    )


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

    def __init__(self, api_key: str | None = None) -> None:
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
        # Tune connection pool to match caller's worker count (Fix OPT-5)
        from requests.adapters import HTTPAdapter
        _adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        self._session.mount("https://", _adapter)

        self._last_request_time: float = 0.0

        # --- Cache path: sanitize and resolve (Fix 11) ----------------------
        raw_path = os.environ.get(_CACHE_PATH_ENV, os.path.join("cache", "hadith_cache"))
        self._cache_path = str(pathlib.Path(raw_path).resolve())
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)

        # --- Open diskcache (Fix 1) -----------------------------------------
        if _DISKCACHE_AVAILABLE:
            self._cache: dict | _diskcache.Cache = _diskcache.Cache(  # type: ignore[type-arg]
                self._cache_path,
                size_limit=_CACHE_SIZE_LIMIT,
            )
            logger.debug("Hadith cache: diskcache at %s", self._cache_path)
        else:
            # Fallback: in-memory dict (no persistence, no TTL, no sharing)
            self._cache = {}
            logger.warning("Hadith cache: in-memory only (diskcache unavailable).")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session and cache."""
        self._session.close()
        if _DISKCACHE_AVAILABLE and isinstance(self._cache, _diskcache.Cache):
            self._cache.close()

    def __enter__(self) -> "HadithMatcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public sync method (unchanged API)
    # ------------------------------------------------------------------

    def match(self, text: str) -> HadithMatch | None:
        """
        Search for *text* in the Sunnah.com hadith database.

        Returns the best HadithMatch above the confidence threshold, or None.
        """
        if not text or not text.strip():
            return None

        query_normalised = normalise_arabic(text)
        candidates = self._cached_search(query_normalised)
        return self._score_candidates(candidates, text, query_normalised)

    # ------------------------------------------------------------------
    # Public async batch method (Fix 9)
    # ------------------------------------------------------------------

    async def match_many_async(self, texts: list[str]) -> list[HadithMatch | None]:
        """
        Asynchronously search for multiple texts using httpx.AsyncClient.

        Falls back to the synchronous path if httpx is not installed.

        Parameters
        ----------
        texts : List of raw Arabic text strings to match.

        Returns
        -------
        List of HadithMatch | None, one per input text, in order.
        """
        if not _HTTPX_AVAILABLE:
            logger.warning("httpx not available — falling back to sync Hadith matching.")
            return [self.match(t) for t in texts]

        headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }

        async def _fetch_one(client: _httpx.AsyncClient, text: str) -> HadithMatch | None:
            if not text or not text.strip():
                return None
            query = normalise_arabic(text)
            if len(query) < 12:
                return None

            # Check cache first (diskcache is sync but fast enough here)
            cached = self._cache_get(query)
            if cached is not None:
                return self._score_candidates(cached, text, query)

            url = f"{SUNNAH_API_BASE}/hadiths/search"
            params = {"q": query, "limit": _SEARCH_LIMIT}

            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    r = await client.get(url, params=params, timeout=10.0)
                    if r.status_code == 200:
                        data = r.json().get("data", [])
                        self._cache_set(query, data)
                        return self._score_candidates(data, text, query)
                    if r.status_code == 429:
                        wait = _BACKOFF_BASE ** attempt
                        logger.warning("Rate limited. Waiting %ds …", int(wait))
                        await asyncio.sleep(wait)
                        continue
                    logger.error("API error %d", r.status_code)
                    return None
                except Exception as exc:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning("Network error (%s). Retry in %ds …", exc, int(wait))
                    await asyncio.sleep(wait)

            return None

        async with _httpx.AsyncClient(headers=headers) as client:
            tasks = [_fetch_one(client, t) for t in texts]
            return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Cache helpers (Fix 1 — diskcache)
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> list[dict] | None:
        """Return cached result or None."""
        try:
            if _DISKCACHE_AVAILABLE and isinstance(self._cache, _diskcache.Cache):
                return self._cache.get(key)
            return self._cache.get(key)  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("Cache read error: %s", exc)
            return None

    def _cache_set(self, key: str, value: list[dict]) -> None:
        """Write to cache with TTL."""
        try:
            if _DISKCACHE_AVAILABLE and isinstance(self._cache, _diskcache.Cache):
                self._cache.set(key, value, expire=_CACHE_TTL_SECONDS)
            else:
                self._cache[key] = value  # type: ignore[index]
        except Exception as exc:
            logger.warning("Cache write error: %s", exc)

    def _cached_search(self, query: str) -> list[dict]:
        """Synchronous cache-then-API lookup."""
        if len(query) < 12:
            return []
        cached = self._cache_get(query)
        if cached is not None:
            return cached
        result = self._search(query)
        self._cache_set(query, result)
        return result

    # ------------------------------------------------------------------
    # Scoring (shared between sync and async paths)
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: list[dict],
        raw_text: str,
        query_normalised: str,
    ) -> HadithMatch | None:
        """Pick the best-scoring candidate above FUZZY_THRESHOLD."""
        best_match: HadithMatch | None = None
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
                    matched_text=raw_text,
                    confidence=score / 100.0,
                )

        if best_match and best_score >= FUZZY_THRESHOLD:
            return best_match
        return None

    # ------------------------------------------------------------------
    # Private HTTP helpers
    # ------------------------------------------------------------------

    def _search(self, query: str) -> list[dict]:
        """Hit the Sunnah.com search endpoint with retry / backoff."""
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
                    return data.get("data", [])

                if response.status_code == 429:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Rate limited. Waiting %ds (attempt %d/%d) …",
                        int(wait), attempt, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

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
