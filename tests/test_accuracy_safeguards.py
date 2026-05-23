from __future__ import annotations

import os

import pytest

from islamic_stt.config import PipelineConfig, apply_profile
from islamic_stt.core.arabic_utils import canonicalise_for_matching
from islamic_stt.core.decoding_passes import segment_needs_retry
from islamic_stt.core.language_detector import detect_language
from islamic_stt.core.post_processor import apply_post_processing
from islamic_stt.core.types import TranscriptSegment
from islamic_stt.matchers.hadith_db import _distinctive_token_count
from islamic_stt.matchers.quran_matcher import (
    QuranMatcher,
    _adaptive_fuzzy_threshold,
    _sliding_word_windows,
)


def test_urdu_exclusive_letters_route_to_urdu():
    text = "یہ پاکستان کی اردو تقریر ہے جس میں حدیث کا ذکر ہے"
    assert detect_language(text) == "ur"


def test_english_pbuh_expands_to_searchable_arabic_formula():
    assert apply_post_processing("PBUH") == "صلى الله عليه وسلم"
    assert apply_post_processing("SAW") == "صلى الله عليه وسلم"


def test_quran_matcher_can_disable_fuzzy_for_low_confidence_text():
    matcher = QuranMatcher(corpus_path="data/quran.json")
    query = "غير المغضوب عليهم ولا الضالين"
    exact_or_fuzzy = matcher.match(query)
    exact_only = matcher.match(query, allow_fuzzy=False)

    assert exact_or_fuzzy is not None
    # This is a partial ayah, so disabling fuzzy should avoid over-attribution.
    assert exact_only is None


def test_generic_hadith_opening_is_not_distinctive_enough():
    generic = "قال رسول الله صلى الله عليه وسلم"
    assert _distinctive_token_count(generic) < 3


def test_colab_accurate_profile_uses_urdu_primary():
    cfg = apply_profile(PipelineConfig(profile="colab-accurate"))
    assert cfg.primary_language == "ur"
    assert cfg.enable_dual_pass_arabic is True
    assert cfg.beam_size == 8


def test_colab_fast_profile_disables_dual_pass():
    cfg = apply_profile(PipelineConfig(profile="colab-fast"))
    assert cfg.enable_dual_pass_arabic is False
    assert cfg.beam_size == 5


def test_adaptive_fuzzy_threshold_lowers_for_moderate_confidence():
    assert _adaptive_fuzzy_threshold(0.80) == 0.88
    assert _adaptive_fuzzy_threshold(0.60) == 0.82


def test_sliding_windows_cover_partial_ayah_fragments():
    query = canonicalise_for_matching("وَلَا تَحْزَنْ إِنَّ اللَّهَ مَعَنَا الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ")
    windows = _sliding_word_windows(query, min_words=5, max_words=8)
    assert len(windows) >= 3
    assert any(len(w.split()) >= 5 for w in windows)


def test_segment_needs_retry_on_low_logprob():
    cfg = PipelineConfig(enable_retry_decoding=True)
    seg = TranscriptSegment(
        id=0,
        start=0.0,
        end=1.0,
        text="test",
        language="ur",
        language_probability=0.9,
        avg_logprob=-1.2,
    )
    assert "low_logprob" in segment_needs_retry(seg, cfg)


@pytest.mark.skipif(
    not os.path.isfile("data/quran.json"),
    reason="Quran corpus required",
)
def test_urdu_script_basmala_matches_quran():
    matcher = QuranMatcher(corpus_path="data/quran.json")
    text = "بسم اللہ الرحمن الرحیم"
    result = matcher.match(text, avg_word_confidence=0.95)
    assert result is not None
    from islamic_stt.matchers.quran_matcher import FormulaMatch

    if isinstance(result, FormulaMatch):
        assert result.label == "Basmala (Urdu script)"
    else:
        assert getattr(result, "surah_id", None) == 1


@pytest.mark.skipif(
    not os.path.isfile("data/quran.json"),
    reason="Quran corpus required",
)
def test_urdu_script_salawat_matches_formula():
    matcher = QuranMatcher(corpus_path="data/quran.json")
    text = "اللہم صلی اللہ علیہ وسلم"
    result = matcher.match(text)
    from islamic_stt.matchers.quran_matcher import FormulaMatch

    assert isinstance(result, FormulaMatch)
