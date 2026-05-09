"""Tests for islamic_stt.core.segment_merger."""

from __future__ import annotations

from islamic_stt.core.segment_merger import merge_short_segments
from islamic_stt.core.transcriber import TranscriptSegment


def _seg(id: int, start: float, end: float, text: str, lang: str = "en") -> TranscriptSegment:
    """Helper to create a minimal TranscriptSegment."""
    return TranscriptSegment(
        id=id, start=start, end=end, text=text,
        language=lang, language_probability=0.9,
    )


class TestMergeShortSegments:
    def test_merge_reduces_count(self):
        segs = [
            _seg(0, 0.0, 1.0, "hello", "en"),
            _seg(1, 1.2, 2.0, "world", "en"),
            _seg(2, 2.3, 3.0, "foo", "en"),
        ]
        merged = merge_short_segments(segs)
        assert len(merged) < len(segs)

    def test_merged_text_contains_both(self):
        segs = [
            _seg(0, 0.0, 1.0, "hello", "en"),
            _seg(1, 1.2, 2.0, "world", "en"),
        ]
        merged = merge_short_segments(segs)
        assert "hello" in merged[0].text
        assert "world" in merged[0].text

    def test_cross_language_not_merged(self):
        segs = [
            _seg(0, 0.0, 1.0, "hi", "en"),
            _seg(1, 1.2, 2.0, "بسم", "ar"),
        ]
        merged = merge_short_segments(segs)
        assert len(merged) == 2, "Cross-language segments must not merge"

    def test_single_segment_passthrough(self):
        segs = [_seg(0, 0.0, 1.0, "hello")]
        merged = merge_short_segments(segs)
        assert len(merged) == 1
        assert merged[0].text == "hello"

    def test_empty_list(self):
        assert merge_short_segments([]) == []

    def test_large_gap_prevents_merge(self):
        segs = [
            _seg(0, 0.0, 1.0, "hi", "en"),
            _seg(1, 5.0, 6.0, "there", "en"),  # 4s gap > threshold
        ]
        merged = merge_short_segments(segs)
        assert len(merged) == 2

    def test_orphan_absorption(self):
        """A long segment followed by a 1-word orphan should merge."""
        segs = [
            _seg(0, 0.0, 2.0, "this is a longer segment with many words", "en"),
            _seg(1, 2.2, 2.5, "yes", "en"),
        ]
        merged = merge_short_segments(segs)
        assert len(merged) == 1
        assert "yes" in merged[0].text
