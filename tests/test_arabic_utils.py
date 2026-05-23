"""Tests for islamic_stt.core.arabic_utils."""

from __future__ import annotations

from islamic_stt.core.arabic_utils import (
    HALLUCINATION_PHRASES_NORMALISED,
    HALLUCINATION_PHRASES_RAW,
    contains_arabic_script,
    normalise_arabic,
    normalise_arabic_cached,
)


class TestNormaliseArabic:
    """normalise_arabic() correctness tests."""

    def test_empty_string(self):
        assert normalise_arabic("") == ""

    def test_latin_passthrough(self):
        assert normalise_arabic("hello world") == "hello world"

    def test_tashkeel_stripping(self):
        # مُحَمَّد → محمد
        assert (
            normalise_arabic("\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f")
            == "\u0645\u062d\u0645\u062f"
        )

    def test_hamza_normalisation(self):
        # ؤ → و, ئ → ي
        assert normalise_arabic("\u0624\u0626") == "\u0648\u064a"

    def test_taa_marbuta_and_alef_maqsura(self):
        # ة → ه, ى → ي
        assert normalise_arabic("\u0629\u0649") == "\u0647\u064a"

    def test_alef_variants(self):
        # أ إ آ ٱ → ا
        result = normalise_arabic("\u0623\u0625\u0622\u0671")
        assert all(c == "\u0627" or c == " " for c in result)

    def test_idempotent(self):
        text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
        once = normalise_arabic(text)
        twice = normalise_arabic(once)
        assert once == twice

    def test_whitespace_collapse(self):
        assert normalise_arabic("  hello   world  ") == "hello world"


class TestCachedNormalisation:
    def test_cached_matches_uncached(self):
        text = "بسم الله"
        assert normalise_arabic_cached(text) == normalise_arabic(text)


class TestContainsArabicScript:
    def test_english_only(self):
        assert contains_arabic_script("hello world") is False

    def test_arabic_present(self):
        assert contains_arabic_script("بسم") is True

    def test_empty_string(self):
        assert contains_arabic_script("") is False

    def test_none_safety(self):
        assert contains_arabic_script(None) is False


class TestHallucinationBlocklist:
    def test_blocklist_not_empty(self):
        assert len(HALLUCINATION_PHRASES_RAW) > 0

    def test_normalised_blocklist_same_size(self):
        # Normalisation may collapse some phrases, but should never increase count
        assert len(HALLUCINATION_PHRASES_NORMALISED) <= len(HALLUCINATION_PHRASES_RAW)
