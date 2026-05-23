"""
diagnostics.py
--------------
Per-segment diagnostic logging for empirical threshold calibration.

Writes one JSON line per segment to diagnostics.jsonl, capturing:
  - confidence signals (avg_logprob, no_speech_prob, word confidence)
  - language detection results and locking decisions
  - prompt mode used
  - match results (Quran, Hadith)
  - correction provenance (raw_text vs final_text)
  - overlap dedup events

This data enables empirical calibration of thresholds instead of guesswork.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["DiagnosticsLogger", "SegmentDiagnostics"]


@dataclass
class SegmentDiagnostics:
    """Diagnostic data for a single segment."""

    segment_id: int
    start: float
    end: float
    text_preview: str  # first 60 chars
    original_text_preview: str  # raw ASR (first 60 chars)
    detected_lang: str
    language_locked: bool = False
    lock_reason: str = ""
    confidence: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    avg_word_confidence: float = 0.0
    word_count: int = 0
    prompt_mode: str = "bayan"
    quran_match: bool = False
    quran_confidence: float = 0.0
    quran_is_exact: bool = False
    quran_is_ambiguous: bool = False
    hadith_match: bool = False
    hadith_confidence: float = 0.0
    formula_match: bool = False
    was_corrected: bool = False
    is_flagged: bool = False
    overlap_removed_words: int = 0
    mixed_script_ratio: float = 0.0
    repeated_ngram_score: float = 0.0
    retry_triggered: bool = False
    retry_reason: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    # Hadith verification diagnostics (Stage 1: suggestion only)
    hadith_retrieval_attempted: bool = False
    hadith_retrieval_accepted: bool = False
    hadith_retrieval_rejected_reason: str = ""
    hadith_verification_confidence: float = 0.0
    hadith_suggestion_generated: bool = False


class DiagnosticsLogger:
    """Writes per-segment diagnostics to a JSONL file."""

    def __init__(self, output_path: str | None = None):
        self.output_path = output_path
        self._entries: list[dict[str, Any]] = []
        self._enabled = output_path is not None

    def add(self, diag: SegmentDiagnostics) -> None:
        """Add a segment diagnostic entry."""
        if not self._enabled:
            return
        self._entries.append(asdict(diag))

    def write(self) -> None:
        """Flush all entries to the JSONL file."""
        if not self._enabled or not self._entries or not self.output_path:
            return

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "Diagnostics: %d segments → %s",
            len(self._entries),
            self.output_path,
        )

    def summarize(self) -> dict[str, Any]:
        """Compute aggregate statistics from collected diagnostics."""
        if not self._entries:
            return {}

        n = len(self._entries)
        confidences = [e["confidence"] for e in self._entries]
        logprobs = [e["avg_logprob"] for e in self._entries]
        confidences.sort()
        logprobs.sort()

        # Language distribution
        lang_counts: dict[str, int] = {}
        for e in self._entries:
            lang = e["detected_lang"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Language drift events (consecutive segments with different langs)
        drift_events = 0
        for i in range(1, n):
            if self._entries[i]["detected_lang"] != self._entries[i - 1]["detected_lang"]:
                drift_events += 1

        return {
            "total_segments": n,
            "confidence_p25": confidences[n // 4] if n >= 4 else confidences[0],
            "confidence_p50": confidences[n // 2],
            "confidence_p75": confidences[3 * n // 4] if n >= 4 else confidences[-1],
            "avg_logprob_p50": logprobs[n // 2],
            "language_distribution": lang_counts,
            "language_drift_events": drift_events,
            "language_locked_count": sum(1 for e in self._entries if e["language_locked"]),
            "quran_matches": sum(1 for e in self._entries if e["quran_match"]),
            "hadith_matches": sum(1 for e in self._entries if e["hadith_match"]),
            "corrected_segments": sum(1 for e in self._entries if e["was_corrected"]),
            "flagged_segments": sum(1 for e in self._entries if e["is_flagged"]),
            "overlap_dedup_total_words": sum(e["overlap_removed_words"] for e in self._entries),
        }
