"""
test_hadith_verifier.py
-----------------------
Comprehensive tests for the retrieval-assisted Hadith verification layer.

Covers:
1. Narration marker detection
2. Normalization correctness
3. Token-level divergence analysis
4. Retrieval confidence scoring
5. Safety gate evaluation
6. Suggestion generation
7. End-to-end verification flow
"""

from __future__ import annotations

import pytest

from islamic_stt.matchers.hadith_verifier import (
    ASR_CONFIDENCE_CEILING,
    MAX_LENGTH_DELTA_RATIO,
    RETRIEVAL_SIMILARITY_FLOOR,
    TOKEN_OVERLAP_FLOOR,
    WORD_CONFIDENCE_FLOOR,
    HadithVerifier,
    TokenDivergence,
    _build_suggestion,
    _unique_marker_categories,
    compute_retrieval_score,
    compute_token_divergence,
    detect_hadith_candidates,
    evaluate_suggestion_safety,
    normalize_for_comparison,
)

# =========================================================================
# 1. Narration marker detection
# =========================================================================


class TestDetectHadithCandidates:
    """Test narration marker detection."""

    def test_prophet_speech_marker(self):
        """Should detect 'قال رسول الله' (the Prophet said)."""
        text = "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert "prophet_speech" in categories
        assert len(hits) >= 1

    def test_narrator_chain_abu_hurayra(self):
        """Should detect 'عن أبي هريرة' (from Abu Hurayra)."""
        text = "عن أبي هريرة رضي الله عنه قال قال رسول الله"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert "narrator_chain" in categories

    def test_transmission_term_haddathana(self):
        """Should detect 'حدثنا' (he narrated to us)."""
        text = "حدثنا محمد بن المثنى قال حدثنا عبد الوهاب"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert "transmission" in categories

    def test_salawat_marker(self):
        """Should detect 'صلى الله عليه وسلم'."""
        text = "النبي صلى الله عليه وسلم"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert "salawat" in categories

    def test_urdu_narrator_marker(self):
        """Should detect Urdu-script narrator names like حضرت ابوہریرہ."""
        text = "حضرت ابوہریرہ سے روایت ہے کہ نبی کریم نے فرمایا"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert "urdu_narrator" in categories or "urdu_prophet" in categories

    def test_urdu_salawat_marker(self):
        """Should detect Urdu-script salawat."""
        text = "رسول اللہ صلی اللہ علیہ وسلم نے فرمایا"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert len(categories) >= 1

    def test_collection_reference(self):
        """Should detect Hadith collection references."""
        text = "صحیح بخاری میں یہ حدیث ہے"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert "collection_ref" in categories

    def test_multiple_markers_in_one_segment(self):
        """A full isnad should produce multiple marker categories."""
        text = "عن أبي هريرة رضي الله عنه قال قال رسول الله صلى الله عليه وسلم"
        hits = detect_hadith_candidates(text)
        categories = _unique_marker_categories(hits)
        assert len(categories) >= 2  # narrator_chain + salawat or prophet_speech

    def test_empty_text_returns_empty(self):
        """Empty input should return no markers."""
        assert detect_hadith_candidates("") == []
        assert detect_hadith_candidates("   ") == []

    def test_non_hadith_text_returns_empty(self):
        """Plain Urdu/English text should return no markers."""
        assert detect_hadith_candidates("آج کا موسم بہت اچھا ہے") == []
        assert detect_hadith_candidates("Today is a good day") == []

    def test_short_text_returns_empty(self):
        """Very short text should return no markers."""
        assert detect_hadith_candidates("بسم") == []


# =========================================================================
# 2. Normalization
# =========================================================================


class TestNormalization:
    """Test Arabic/Urdu normalization for comparison."""

    def test_basic_arabic_normalization(self):
        """Should strip diacritics and normalize alef."""
        result = normalize_for_comparison("بِسْمِ اللَّهِ")
        assert "بسم" in result
        assert "الله" in result or "اللہ" in result

    def test_empty_input(self):
        assert normalize_for_comparison("") == ""

    def test_idempotent(self):
        """Normalizing twice should give the same result."""
        text = "قال رسول الله صلى الله عليه وسلم"
        first = normalize_for_comparison(text)
        second = normalize_for_comparison(first)
        assert first == second

    def test_cross_script_urdu_to_arabic(self):
        """Urdu script characters should map to Arabic equivalents."""
        # ہ (Urdu Heh Goal) → ه (Arabic Heh)
        # ی (Urdu Yeh) → ي (Arabic Yeh)
        urdu = "اللہ کی"
        arabic = "الله كي"
        urdu_norm = normalize_for_comparison(urdu)
        arabic_norm = normalize_for_comparison(arabic)
        # After normalization, they should be identical
        assert urdu_norm == arabic_norm


# =========================================================================
# 3. Token-level divergence
# =========================================================================


class TestTokenDivergence:
    """Test token-level divergence analysis."""

    def test_identical_texts_no_divergence(self):
        """Identical texts should have no divergent tokens."""
        text = "إنما الأعمال بالنيات"
        divergences = compute_token_divergence(text, text)
        assert all(not d.is_divergent for d in divergences)

    def test_single_word_difference(self):
        """One word different should flag that token as divergent."""
        asr = "إنما الاعمل بالنيات"  # الاعمل is wrong
        ret = "إنما الأعمال بالنيات"  # الأعمال is correct
        divergences = compute_token_divergence(asr, ret)
        divergent = [d for d in divergences if d.is_divergent]
        assert len(divergent) >= 1

    def test_word_confidences_attached(self):
        """Word confidences should be attached to divergence results."""
        asr = "قال رسول الله"
        ret = "قال رسول الله"
        confs = [0.9, 0.8, 0.7]
        divergences = compute_token_divergence(asr, ret, confs)
        assert divergences[0].asr_word_confidence == 0.9
        assert divergences[1].asr_word_confidence == 0.8

    def test_missing_confidences_default_to_zero(self):
        """Missing confidences should default to 0.0 (conservative)."""
        asr = "قال رسول"
        ret = "قال رسول"
        divergences = compute_token_divergence(asr, ret)
        assert all(d.asr_word_confidence == 0.0 for d in divergences)

    def test_empty_inputs(self):
        """Empty inputs should return empty list."""
        assert compute_token_divergence("", "some text") == []
        assert compute_token_divergence("some text", "") == []

    def test_similarity_scores_are_valid(self):
        """Similarity scores should be between 0 and 1."""
        divergences = compute_token_divergence("قال رسول الله", "قال نبي الله")
        for d in divergences:
            assert 0.0 <= d.similarity <= 1.0


# =========================================================================
# 4. Retrieval confidence scoring
# =========================================================================


class TestRetrievalScoring:
    """Test retrieval similarity computation."""

    def test_identical_texts_high_score(self):
        """Identical texts should produce near-perfect scores."""
        text = "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"
        overlap, fuzzy, combined = compute_retrieval_score(text, text)
        assert overlap >= 0.99
        assert fuzzy >= 0.99
        assert combined >= 0.95

    def test_completely_different_texts_low_score(self):
        """Unrelated texts should produce low scores."""
        overlap, fuzzy, combined = compute_retrieval_score(
            "الحمد لله رب العالمين", "Today is a beautiful day"
        )
        assert overlap < 0.1
        assert combined < 0.3

    def test_partial_overlap_medium_score(self):
        """Partial overlap should produce intermediate scores."""
        overlap, fuzzy, combined = compute_retrieval_score(
            "إنما الأعمال بالنيات", "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"
        )
        assert 0.3 < overlap < 0.9
        assert combined > 0.5

    def test_empty_inputs_zero_score(self):
        """Empty inputs should return zero scores."""
        assert compute_retrieval_score("", "test") == (0.0, 0.0, 0.0)
        assert compute_retrieval_score("test", "") == (0.0, 0.0, 0.0)

    def test_scores_are_bounded(self):
        """All scores should be between 0 and 1."""
        overlap, fuzzy, combined = compute_retrieval_score(
            "قال رسول الله صلى الله عليه وسلم", "قال النبي صلى الله عليه وسلم"
        )
        assert 0.0 <= overlap <= 1.0
        assert 0.0 <= fuzzy <= 1.0
        assert 0.0 <= combined <= 1.0


# =========================================================================
# 5. Safety gate evaluation
# =========================================================================


class TestSafetyGates:
    """Test safety gate evaluation — each gate independently."""

    def _make_divergent_token(
        self,
        *,
        is_divergent: bool = True,
        asr_word_confidence: float = 0.2,
    ) -> TokenDivergence:
        """Helper to create a TokenDivergence for testing."""
        return TokenDivergence(
            asr_token="الاعمل",
            retrieval_token="الأعمال",
            position=1,
            asr_word_confidence=asr_word_confidence,
            is_divergent=is_divergent,
            similarity=0.5,
        )

    def _passing_args(self, **overrides):
        """Return a dict of args that pass all gates, with overrides."""
        # Both texts are similar enough for length delta check
        base_text = "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات"
        defaults = dict(
            asr_confidence=0.30,  # below ceiling
            retrieval_confidence=0.90,  # above floor
            token_overlap=0.70,  # above floor
            marker_count=2,  # above minimum
            divergent_tokens=[self._make_divergent_token()],
            asr_text=base_text,
            retrieval_text=base_text,
        )
        defaults.update(overrides)
        return defaults

    def test_all_gates_pass(self):
        """When all gates pass, suggestion should be safe."""
        is_safe, reason = evaluate_suggestion_safety(**self._passing_args())
        assert is_safe is True
        assert reason == ""

    def test_gate1_asr_confidence_too_high(self):
        """High ASR confidence should block suggestion."""
        is_safe, reason = evaluate_suggestion_safety(**self._passing_args(asr_confidence=0.80))
        assert is_safe is False
        assert "asr_confidence_high" in reason

    def test_gate2_retrieval_similarity_too_low(self):
        """Low retrieval similarity should block suggestion."""
        is_safe, reason = evaluate_suggestion_safety(
            **self._passing_args(retrieval_confidence=0.60)
        )
        assert is_safe is False
        assert "low_retrieval_similarity" in reason

    def test_gate3_token_overlap_too_low(self):
        """Low token overlap should block suggestion."""
        is_safe, reason = evaluate_suggestion_safety(**self._passing_args(token_overlap=0.40))
        assert is_safe is False
        assert "low_token_overlap" in reason

    def test_gate4_no_narration_markers(self):
        """No narration markers should block suggestion."""
        is_safe, reason = evaluate_suggestion_safety(**self._passing_args(marker_count=0))
        assert is_safe is False
        assert "no_narration_markers" in reason

    def test_gate5_span_exceeds_limit(self):
        """Too many divergent tokens should block suggestion."""
        # Create 3 divergent tokens (exceeds limit of 2)
        tokens = [self._make_divergent_token() for _ in range(3)]
        is_safe, reason = evaluate_suggestion_safety(**self._passing_args(divergent_tokens=tokens))
        assert is_safe is False
        assert "span_exceeds_limit" in reason

    def test_gate6_length_delta_too_high(self):
        """Large length difference should block suggestion."""
        short_text = "قال رسول الله"
        long_text = "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"
        is_safe, reason = evaluate_suggestion_safety(
            **self._passing_args(asr_text=short_text, retrieval_text=long_text)
        )
        assert is_safe is False
        assert "length_delta_high" in reason

    def test_gate7_acoustic_divergent_token_high_confidence(self):
        """Divergent token with high ASR confidence should block suggestion."""
        # ASR was confident about a word that differs from retrieval
        token = self._make_divergent_token(
            is_divergent=True,
            asr_word_confidence=0.85,  # above ceiling
        )
        is_safe, reason = evaluate_suggestion_safety(**self._passing_args(divergent_tokens=[token]))
        assert is_safe is False
        assert "divergent_token_high_confidence" in reason

    def test_non_divergent_tokens_dont_trigger_acoustic_gate(self):
        """Non-divergent tokens should not trigger the acoustic gate."""
        token = TokenDivergence(
            asr_token="الأعمال",
            retrieval_token="الأعمال",
            position=1,
            asr_word_confidence=0.95,  # high but NOT divergent
            is_divergent=False,
            similarity=1.0,
        )
        is_safe, reason = evaluate_suggestion_safety(**self._passing_args(divergent_tokens=[token]))
        # Should pass because there are no divergent tokens
        assert is_safe is True


# =========================================================================
# 6. Suggestion generation
# =========================================================================


class TestSuggestionGeneration:
    """Test suggestion string building."""

    def test_build_suggestion_with_divergent_tokens(self):
        """Should produce 'old→new' format for divergent tokens."""
        tokens = [
            TokenDivergence("الاعمل", "الأعمال", 1, 0.2, True, 0.5),
        ]
        result = _build_suggestion(tokens)
        assert "الاعمل→الأعمال" in result

    def test_build_suggestion_no_divergence(self):
        """No divergent tokens should produce empty string."""
        tokens = [
            TokenDivergence("الأعمال", "الأعمال", 1, 0.9, False, 1.0),
        ]
        result = _build_suggestion(tokens)
        assert result == ""

    def test_build_suggestion_multiple_divergences(self):
        """Multiple divergences should be comma-separated."""
        tokens = [
            TokenDivergence("كلمة١", "كلمة٢", 0, 0.2, True, 0.4),
            TokenDivergence("كلمة٣", "كلمة٤", 1, 0.3, True, 0.5),
        ]
        result = _build_suggestion(tokens)
        assert "," in result
        assert "كلمة١→كلمة٢" in result
        assert "كلمة٣→كلمة٤" in result


# =========================================================================
# 7. End-to-end verification
# =========================================================================


class TestHadithVerifier:
    """Test the full HadithVerifier.verify() flow."""

    @pytest.fixture
    def verifier(self):
        return HadithVerifier()

    def test_applied_is_always_false(self, verifier):
        """Stage 1: applied must always be False."""
        result = verifier.verify(
            segment_id=1,
            asr_text="قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات",
            retrieval_text="قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات",
            retrieval_collection="bukhari",
            retrieval_number="1",
            retrieval_confidence=0.95,
            asr_word_confidence=0.30,
        )
        assert result.applied is False  # CRITICAL: Stage 1 invariant

    def test_high_confidence_asr_no_suggestion(self, verifier):
        """High ASR confidence should not produce safe suggestions."""
        result = verifier.verify(
            segment_id=2,
            asr_text="قال رسول الله صلى الله عليه وسلم من حسن إسلام المرء تركه ما لا يعنيه",
            retrieval_text="قال رسول الله صلى الله عليه وسلم من حسن إسلام المرء تركه ما لا يعنيه",
            retrieval_collection="tirmidhi",
            retrieval_number="2317",
            retrieval_confidence=0.90,
            asr_word_confidence=0.82,  # HIGH confidence — trust ASR
        )
        assert result.suggestion_safe is False
        assert result.applied is False

    def test_low_asr_high_retrieval_generates_suggestion(self, verifier):
        """Low ASR + high retrieval + markers should generate suggestion."""
        asr = "قال رسول الله صلى الله عليه وسلم إنما الاعمل بالنيات"
        ret = "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات"
        result = verifier.verify(
            segment_id=3,
            asr_text=asr,
            retrieval_text=ret,
            retrieval_collection="bukhari",
            retrieval_number="1",
            retrieval_confidence=0.95,
            asr_word_confidence=0.25,  # LOW confidence
            word_confidences=[0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.5, 0.15, 0.6],
        )
        # Should have detected markers and generated a suggestion
        assert result.narration_marker_count >= 1
        assert result.applied is False  # Stage 1 invariant

    def test_verification_result_has_all_metadata(self, verifier):
        """Verification result should have all required metadata fields."""
        result = verifier.verify(
            segment_id=4,
            asr_text="عن أبي هريرة رضي الله عنه",
            retrieval_text="عن أبي هريرة رضي الله عنه",
            retrieval_collection="bukhari",
            retrieval_number="1",
            retrieval_confidence=0.90,
            asr_word_confidence=0.50,
        )
        # Check all metadata fields exist
        assert hasattr(result, "segment_id")
        assert hasattr(result, "retrieval_confidence")
        assert hasattr(result, "token_overlap")
        assert hasattr(result, "fuzzy_similarity")
        assert hasattr(result, "narration_marker_count")
        assert hasattr(result, "marker_categories")
        assert hasattr(result, "divergent_tokens")
        assert hasattr(result, "suggested_patch")
        assert hasattr(result, "suggestion_safe")
        assert hasattr(result, "applied")
        assert hasattr(result, "correction_type")
        assert hasattr(result, "rejection_reason")

    def test_no_markers_blocks_suggestion(self, verifier):
        """Text without narration markers should not produce safe suggestion."""
        result = verifier.verify(
            segment_id=5,
            asr_text="الحمد لله رب العالمين",  # Quran, not Hadith
            retrieval_text="الحمد لله رب العالمين",
            retrieval_collection="unknown",
            retrieval_number="0",
            retrieval_confidence=0.95,
            asr_word_confidence=0.20,
        )
        # No narration markers → suggestion should not be safe
        assert result.narration_marker_count == 0
        assert result.suggestion_safe is False

    def test_correction_type_values(self, verifier):
        """correction_type should be one of 'suggestion', 'none', 'rejected'."""
        result = verifier.verify(
            segment_id=6,
            asr_text="قال رسول الله صلى الله عليه وسلم",
            retrieval_text="قال رسول الله صلى الله عليه وسلم",
            retrieval_collection="bukhari",
            retrieval_number="1",
            retrieval_confidence=0.90,
            asr_word_confidence=0.50,
        )
        assert result.correction_type in ("suggestion", "none", "rejected")


# =========================================================================
# 8. Threshold sanity checks
# =========================================================================


class TestThresholdValues:
    """Verify threshold constants are set to safe values."""

    def test_asr_ceiling_is_conservative(self):
        """ASR ceiling should be low — only correct uncertain segments."""
        assert ASR_CONFIDENCE_CEILING <= 0.50

    def test_retrieval_floor_is_high(self):
        """Retrieval floor should be high — only trust strong matches."""
        assert RETRIEVAL_SIMILARITY_FLOOR >= 0.80

    def test_token_overlap_floor_is_reasonable(self):
        """Token overlap should require substantial overlap."""
        assert TOKEN_OVERLAP_FLOOR >= 0.50

    def test_length_delta_is_conservative(self):
        """Length delta should not allow large expansions."""
        assert MAX_LENGTH_DELTA_RATIO <= 0.35

    def test_word_confidence_floor_is_low(self):
        """Word confidence floor marks genuinely uncertain words."""
        assert WORD_CONFIDENCE_FLOOR <= 0.40


# =========================================================================
# 9. Structural Safety Tests
# =========================================================================


class TestStructuralMismatch:
    """Test that structural mismatches are rejected."""

    def test_deletion_rejected(self):
        """If ASR has extra words that are deleted in retrieval, it is unsafe."""
        # ASR has "من" at index 2
        asr = "إنما الأعمال من بالنيات"
        ret = "إنما الأعمال بالنيات"
        divergences = compute_token_divergence(asr, ret)
        # Find the deletion divergence
        deleted = [d for d in divergences if d.asr_token == "من"]
        assert len(deleted) == 1
        assert deleted[0].retrieval_token == ""
        assert deleted[0].is_divergent is True

        # Safety gate should reject it
        is_safe, reason = evaluate_suggestion_safety(
            asr_confidence=0.30,
            retrieval_confidence=0.90,
            token_overlap=0.75,
            marker_count=2,
            divergent_tokens=divergences,
            asr_text=asr,
            retrieval_text=ret,
        )
        assert is_safe is False
        assert "structural_mismatch" in reason

    def test_block_size_mismatch_rejected(self):
        """Replacing 1 word with 2 words is a structural mismatch and is unsafe."""
        asr = "إنما الاعمل بالنيات"
        # "الاعمل" (1 word) -> "الأعمال وأما" (2 words)
        ret = "إنما الأعمال وأما بالنيات"
        divergences = compute_token_divergence(asr, ret)
        # Find the mismatch divergence
        mismatched = [d for d in divergences if d.asr_token == "الاعمل"]
        assert len(mismatched) == 1
        assert mismatched[0].retrieval_token == "الاعمال واما"
        assert mismatched[0].is_divergent is True

        # Safety gate should reject it
        is_safe, reason = evaluate_suggestion_safety(
            asr_confidence=0.30,
            retrieval_confidence=0.90,
            token_overlap=0.75,
            marker_count=2,
            divergent_tokens=divergences,
            asr_text=asr,
            retrieval_text=ret,
        )
        assert is_safe is False
        assert "structural_mismatch" in reason
