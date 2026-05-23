"""
segment_merger.py
-----------------
Merges short Whisper fragments into longer, more readable transcript blocks.

Problem
-------
Whisper large-v3 frequently produces very short segments (1–3 words) in
lecture audio, especially during rapid speech, filler words, or when the
speaker pauses mid-sentence.  This results in transcripts like:

    [00:00:35] چیز کی عبادات جو کروں سجود
    [00:00:37] کریں اس کے بعد بس ہماری
    [00:00:40] مال آ جائے ہمارے پاس

This module merges consecutive short segments into coherent blocks when
the gap between them is small, dramatically improving readability.

Rules
-----
- Segments with fewer than MIN_SEGMENT_WORDS words are merge candidates.
- Only merge when the gap between segments < MAX_MERGE_GAP_SECONDS.
- Never merge across different detected languages (important for
  Urdu/Arabic/English mixed content).
- Cap merged segment duration at MAX_MERGED_DURATION_SECONDS to keep
  segments suitable for subtitle use.
- Preserve word-level timestamps by concatenating word lists.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from islamic_stt.core.arabic_utils import contains_arabic_script
from islamic_stt.core.types import TranscriptSegment

__all__ = ["merge_short_segments", "merge_arabic_quote_blocks"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Segments with fewer words than this are candidates for merging with
# their neighbours.
MIN_SEGMENT_WORDS = 4

# Maximum silence gap (in seconds) between two segments that can be merged.
# Beyond this, we assume a deliberate pause / topic change.
MAX_MERGE_GAP_SECONDS = 1.5

# Upper bound on the duration of a merged segment.  Keeps output suitable
# for subtitles (SRT/VTT) where very long cues are unreadable.
MAX_MERGED_DURATION_SECONDS = 30.0

# Second pass: merge Urdu segments containing Arabic quotes (ayah fragments).
ARABIC_QUOTE_MAX_GAP_SECONDS = 2.0
ARABIC_QUOTE_MAX_DURATION_SECONDS = 45.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_short_segments(
    segments: list[TranscriptSegment],
    *,
    min_words: int = MIN_SEGMENT_WORDS,
    max_gap: float = MAX_MERGE_GAP_SECONDS,
    max_duration: float = MAX_MERGED_DURATION_SECONDS,
) -> list[TranscriptSegment]:
    """
    Merge consecutive short segments into longer, coherent blocks.

    Parameters
    ----------
    segments     : Raw segment list from the transcriber.
    min_words    : Segments with fewer words are merge candidates.
    max_gap      : Maximum gap (seconds) to allow merging across.
    max_duration : Maximum duration (seconds) for a merged segment.

    Returns
    -------
    New list with fewer, longer segments.  Original list is not mutated.
    """
    if len(segments) <= 1:
        return list(segments)

    merged: list[TranscriptSegment] = []
    current = segments[0]

    for seg in segments[1:]:
        gap = seg.start - current.end
        current_word_count = len(current.text.split())
        next_word_count = len(seg.text.split())
        merged_duration = seg.end - current.start

        can_merge = (
            (current_word_count < min_words or next_word_count < min_words)
            and gap < max_gap
            and gap >= 0.0  # reject overlapping-backwards
            and seg.language == current.language  # never merge across languages
            and merged_duration < max_duration
        )

        # Micro-fragment merge: single-word fragments with small gap
        # are merged more aggressively for readability, but with
        # confidence guards to avoid merging across speaker transitions.
        if not can_merge and (current_word_count < 2 or next_word_count < 2):
            seg_gap_small = gap < 0.7 and gap >= 0.0
            same_lang = seg.language == current.language
            confidence_close = abs(seg.avg_logprob - current.avg_logprob) < 0.5
            micro_duration = merged_duration < max_duration
            can_merge = seg_gap_small and same_lang and confidence_close and micro_duration

        if can_merge:
            # Merge seg into current
            current = _merge_pair(current, seg)
        else:
            merged.append(current)
            current = seg

    merged.append(current)

    reduction = len(segments) - len(merged)
    if reduction > 0:
        logger.info(
            "Segment merger: %d → %d segments (%d merged)",
            len(segments),
            len(merged),
            reduction,
        )
    return merged


def merge_arabic_quote_blocks(
    segments: list[TranscriptSegment],
    *,
    max_gap: float = ARABIC_QUOTE_MAX_GAP_SECONDS,
    max_duration: float = ARABIC_QUOTE_MAX_DURATION_SECONDS,
) -> list[TranscriptSegment]:
    """
    Merge consecutive Urdu segments that contain Arabic-script quotes
    so partial ayah matching sees longer contiguous text.
    """
    if len(segments) <= 1:
        return list(segments)

    merged: list[TranscriptSegment] = []
    current = segments[0]

    for seg in segments[1:]:
        gap = seg.start - current.end
        duration = seg.end - current.start
        both_urdu = current.language == "ur" and seg.language == "ur"
        both_arabic_script = contains_arabic_script(current.text) and contains_arabic_script(
            seg.text
        )
        can_merge = (
            both_urdu
            and both_arabic_script
            and gap < max_gap
            and gap >= 0.0
            and duration < max_duration
        )
        if can_merge:
            current = _merge_pair(current, seg)
        else:
            merged.append(current)
            current = seg

    merged.append(current)
    if len(merged) < len(segments):
        logger.info(
            "Arabic quote merger: %d → %d segments.",
            len(segments),
            len(merged),
        )
    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_pair(a: TranscriptSegment, b: TranscriptSegment) -> TranscriptSegment:
    """
    Merge segment *b* into *a*, producing a new segment that spans both.

    Uses ``dataclasses.replace`` to avoid mutating the originals.
    """
    combined_words = a.words + b.words
    avg_wc = (
        sum(w.probability for w in combined_words) / len(combined_words) if combined_words else 0.0
    )

    return replace(
        a,
        end=b.end,
        text=f"{a.text} {b.text}",
        words=combined_words,
        avg_logprob=(a.avg_logprob + b.avg_logprob) / 2,
        no_speech_prob=max(a.no_speech_prob, b.no_speech_prob),
        avg_word_confidence=avg_wc,
    )
