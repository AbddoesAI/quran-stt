"""
overlap_merger.py
-----------------
Semantic overlap deduplication for chunk-boundary artifacts.

When using chunk overlap (e.g. 2s), Whisper produces duplicate text
at chunk boundaries. This module detects and removes those duplicates
using normalized Arabic comparison.

Strategy:
  1. Detect time-overlapping segments (seg_b.start < seg_a.end)
  2. Normalize both texts for comparison (strip diacritics, canonicalize)
  3. Find longest common suffix(A) / prefix(B)
  4. Remove the duplicate prefix from B
  5. Preserve original (non-normalized) text in output

Safety:
  - max_overlap_words = 15 to prevent accidental large merges
  - Comparison uses normalise_arabic() for orthographic canonicalization
"""

from __future__ import annotations

import logging
from dataclasses import replace

from islamic_stt.core.arabic_utils import normalise_arabic
from islamic_stt.core.types import TranscriptSegment

logger = logging.getLogger(__name__)

__all__ = ["deduplicate_overlap"]

# Maximum words to consider for overlap matching.
# Prevents accidental large merges on repeated Quranic content.
MAX_OVERLAP_WORDS = 15


def deduplicate_overlap(
    segments: list[TranscriptSegment],
    overlap_seconds: float = 2.0,
) -> list[TranscriptSegment]:
    """
    Remove duplicated text at chunk boundaries.

    For consecutive segments where seg_b.start < seg_a.end (time overlap):
      1. Normalize last N words of A and first N words of B
      2. Find longest matching suffix/prefix
      3. Remove the duplicate from B

    Parameters
    ----------
    segments        : List of segments in time order.
    overlap_seconds : Maximum time overlap to consider (matches chunk_overlap).

    Returns
    -------
    Deduplicated segment list (may modify segment text in-place).
    """
    if len(segments) < 2:
        return segments

    dedup_count = 0

    for i in range(len(segments) - 1):
        seg_a = segments[i]
        seg_b = segments[i + 1]

        # Check for time overlap
        if seg_b.start >= seg_a.end:
            continue  # no overlap

        time_overlap = seg_a.end - seg_b.start
        if time_overlap > overlap_seconds + 0.5:
            continue  # too large — not a chunk boundary

        # Get words from each segment
        words_a_raw = seg_a.text.split()
        words_b_raw = seg_b.text.split()

        if not words_a_raw or not words_b_raw:
            continue

        # Take last/first N words for comparison
        n = min(MAX_OVERLAP_WORDS, len(words_a_raw), len(words_b_raw))
        tail_a_raw = words_a_raw[-n:]
        head_b_raw = words_b_raw[:n]

        # Normalize for comparison (strip diacritics, canonicalize)
        # ٱ→ا, أ→ا, إ→ا, آ→ا, ى→ي, ة→ه, strip harakat
        tail_a_norm = [normalise_arabic(w) for w in tail_a_raw]
        head_b_norm = [normalise_arabic(w) for w in head_b_raw]

        # Find longest matching suffix of A / prefix of B
        best_overlap = 0
        for overlap_len in range(min(n, len(tail_a_norm), len(head_b_norm)), 0, -1):
            if tail_a_norm[-overlap_len:] == head_b_norm[:overlap_len]:
                best_overlap = overlap_len
                break

        if best_overlap > 0:
            # Remove the duplicate prefix from B (using raw words, not normalized)
            remaining_words = words_b_raw[best_overlap:]
            if remaining_words:
                new_text = " ".join(remaining_words)
                segments[i + 1] = replace(seg_b, text=new_text)
                dedup_count += 1
                logger.debug(
                    "Overlap dedup: removed %d words from segment %d "
                    "(%.1fs-%.1fs overlap with segment %d)",
                    best_overlap,
                    seg_b.id,
                    seg_b.start,
                    seg_a.end,
                    seg_a.id,
                )
            else:
                # Entire segment B was a duplicate — mark as empty
                segments[i + 1] = replace(seg_b, text="")
                dedup_count += 1
                logger.debug(
                    "Overlap dedup: entire segment %d was duplicate of %d",
                    seg_b.id,
                    seg_a.id,
                )

    if dedup_count > 0:
        # Remove empty segments
        segments = [s for s in segments if s.text.strip()]
        logger.info("Overlap dedup: removed %d duplicate(s).", dedup_count)

    return segments
