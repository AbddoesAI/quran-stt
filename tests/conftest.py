"""Shared pytest fixtures for the Islamic STT test suite."""

from __future__ import annotations

import pytest

from islamic_stt.core.transcriber import TranscriptSegment


@pytest.fixture
def sample_segment() -> TranscriptSegment:
    """A minimal TranscriptSegment for unit tests."""
    return TranscriptSegment(
        id=0,
        start=0.0,
        end=1.0,
        text="test segment",
        language="en",
        language_probability=0.95,
    )


@pytest.fixture
def arabic_segment() -> TranscriptSegment:
    """An Arabic TranscriptSegment for matcher tests."""
    return TranscriptSegment(
        id=1,
        start=1.0,
        end=3.0,
        text="بسم الله الرحمن الرحيم",
        language="ar",
        language_probability=0.99,
    )
