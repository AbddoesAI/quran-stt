"""
types.py
--------
Shared data structures used across the pipeline.

Extracted to break the circular import between transcriber.py and
segment_merger.py (P0 import cycle fix).

All modules that need TranscriptSegment or WordTimestamp should import
from here, not from transcriber.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TranscriptSegment", "WordTimestamp"]


@dataclass(slots=True)
class WordTimestamp:
    """A single transcribed word with its timing and confidence."""

    word: str
    start: float
    end: float
    probability: float


@dataclass(slots=True)
class TranscriptSegment:
    """
    One contiguous speech segment as returned by faster-whisper.

    Fields
    ------
    id          : zero-based segment index
    start/end   : segment boundaries in seconds
    text        : raw transcribed text for this segment
    language    : ISO-639-1 code — this is the MODEL-LEVEL language
                  (file-level or forced). Do NOT use for segment routing.
                  Use `detected_language` from EnrichedSegment instead.
    language_probability : Whisper's confidence in the file-level language.
    avg_logprob : average log-probability of tokens in this segment.
    words       : word-level timestamps (populated when word_timestamps=True)
    no_speech_prob : probability that the segment contains no speech.
    avg_word_confidence : mean of word-level probabilities (0.0–1.0).
    """

    id: int
    start: float
    end: float
    text: str
    language: str | None
    language_probability: float
    avg_logprob: float = 0.0
    words: list[WordTimestamp] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_word_confidence: float = 0.0
    compression_ratio: float = 0.0
    mixed_script_ratio: float = 0.0
    repeated_ngram_score: float = 0.0
    prompt_safe: bool = False
    retry_triggered: bool = False
    retry_reason: list[str] = field(default_factory=list)
    original_text: str | None = None
    confidence_delta: float = 0.0
