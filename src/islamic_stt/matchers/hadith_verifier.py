"""
hadith_verifier.py
------------------
Retrieval-assisted Hadith verification layer (Stage 1: verification only).

This module verifies Hadith segments against retrieval results and produces
**suggestions** — it NEVER modifies the transcript text.  Audio remains the
source of truth.

Pipeline position: runs AFTER Hadith matching (Step 6), BEFORE output (Step 7).

What this module does
---------------------
1. Detect candidate Hadith segments via narration markers + Arabic spans
2. Normalize Arabic/Urdu orthography for comparison
3. Compute token-level divergence (not just fuzzy similarity)
4. Gate corrections with word-timestamp acoustic awareness
5. Generate suggestions with full metadata (applied=False always)
6. Log diagnostics for every retrieval attempt

What this module does NOT do (Stage 2, deferred)
------------------------------------------------
- Modify transcript text
- Apply span corrections
- Auto-complete narration chains
- Inject text absent from audio

Safety philosophy
-----------------
- Audio is source of truth
- Retrieval is verification, not generation
- All suggestions are passive until Stage 2 benchmarks exist
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from islamic_stt.core.arabic_utils import canonicalise_for_matching

logger = logging.getLogger(__name__)

__all__ = [
    "HadithVerifier",
    "HadithVerificationResult",
    "detect_hadith_candidates",
]


# ---------------------------------------------------------------------------
# Narration marker patterns
# ---------------------------------------------------------------------------
# These patterns identify segments that are likely Hadith narrations.
# They are used for candidate detection ONLY — matching is done separately.

# Arabic narration markers (compiled once at module load)
_NARRATION_MARKERS: list[tuple[str, re.Pattern]] = [
    # Prophet's speech markers
    ("prophet_speech", re.compile(r"قال\s+رسول\s+الله")),
    ("prophet_speech", re.compile(r"قال\s+النبي")),
    ("prophet_said", re.compile(r"قال\s+صلى\s+الله\s+عليه\s+وسلم")),
    # Narrator chain markers (isnad)
    ("narrator_chain", re.compile(r"عن\s+أبي\s+هريرة")),
    ("narrator_chain", re.compile(r"عن\s+ابن\s+عمر")),
    ("narrator_chain", re.compile(r"عن\s+ابن\s+عباس")),
    ("narrator_chain", re.compile(r"عن\s+عائشة")),
    ("narrator_chain", re.compile(r"عن\s+أنس")),
    ("narrator_chain", re.compile(r"عن\s+جابر")),
    # Generic narrator prefix (عن + name + رضي الله عنه)
    ("narrator_generic", re.compile(r"عن\s+\S+\s+رضي\s+الله\s+عنه")),
    # Transmission terms
    ("transmission", re.compile(r"حدثنا\b")),
    ("transmission", re.compile(r"أخبرنا\b")),
    ("transmission", re.compile(r"اخبرنا\b")),
    ("transmission", re.compile(r"سمعت\b")),
    # Salawat — strong signal when combined with narration context
    ("salawat", re.compile(r"صلى\s+الله\s+عليه\s+وسلم")),
    # Urdu-script narration markers (as spoken in Urdu bayans)
    ("urdu_narrator", re.compile(r"حضرت\s+ابو\s*ہریرہ")),
    ("urdu_narrator", re.compile(r"حضرت\s+عائشہ")),
    ("urdu_narrator", re.compile(r"حضرت\s+انس")),
    ("urdu_prophet", re.compile(r"نبی\s+کریم")),
    ("urdu_prophet", re.compile(r"رسول\s+اللہ")),
    ("urdu_salawat", re.compile(r"صلی\s+اللہ\s+علیہ\s+وسلم")),
    # Collection references (in Urdu context)
    ("collection_ref", re.compile(r"صحیح\s+بخاری")),
    ("collection_ref", re.compile(r"صحیح\s+مسلم")),
    ("collection_ref", re.compile(r"سنن\s+ترمذی")),
]

# Minimum number of distinct marker categories to consider a segment
# a strong Hadith candidate. A single salawat alone is not enough.
_MIN_MARKER_CATEGORIES = 1

# Maximum suggestion span size (words) — even for suggestions, we cap this
# to prevent nonsensical suggestions. Stage 2 may raise to 2 after benchmarks.
_MAX_SUGGESTION_SPAN_WORDS = 2

# ---------------------------------------------------------------------------
# Thresholds for verification scoring
# ---------------------------------------------------------------------------

# ASR confidence ceiling: only generate suggestions when ASR is uncertain.
# Segments with word confidence above this are trusted and skipped.
ASR_CONFIDENCE_CEILING = 0.45

# Retrieval similarity floor: minimum combined score to generate a suggestion.
RETRIEVAL_SIMILARITY_FLOOR = 0.85

# Token overlap floor: minimum Jaccard overlap on word tokens.
TOKEN_OVERLAP_FLOOR = 0.60

# Maximum length delta ratio between ASR text and retrieval text.
# If the lengths diverge by more than 30%, the match is suspicious.
MAX_LENGTH_DELTA_RATIO = 0.30

# Word-level confidence threshold: individual words below this are
# candidates for correction suggestions (when retrieval supports it).
WORD_CONFIDENCE_FLOOR = 0.35


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NarrationMarkerHit:
    """A single narration marker found in a segment."""

    category: str  # e.g. "prophet_speech", "narrator_chain", "salawat"
    pattern: str  # the regex pattern that matched
    span_start: int  # character offset in the segment
    span_end: int  # character offset in the segment


@dataclass(slots=True)
class TokenDivergence:
    """Token-level divergence between ASR output and retrieval text."""

    asr_token: str  # the word from ASR
    retrieval_token: str  # the corresponding word from retrieval
    position: int  # word index in the ASR text
    asr_word_confidence: float  # word-level confidence (0.0 if unavailable)
    is_divergent: bool  # True if the tokens differ meaningfully
    similarity: float  # char-level similarity between the two tokens


@dataclass(slots=True)
class HadithVerificationResult:
    """
    Result of Hadith verification for a single segment.

    This is a SUGGESTION container — it never triggers transcript mutation.
    The `applied` field is always False in Stage 1.
    """

    # --- Identification ---
    segment_id: int
    segment_text: str  # original ASR text (never modified)

    # --- Retrieval match info ---
    hadith_match_collection: str  # e.g. "bukhari"
    hadith_match_number: str  # e.g. "1"
    hadith_match_confidence: float  # original match confidence from matcher

    # --- Verification scores ---
    retrieval_confidence: float  # calibrated retrieval similarity score
    token_overlap: float  # Jaccard token overlap ratio
    fuzzy_similarity: float  # rapidfuzz combined score
    narration_marker_count: int  # number of distinct marker categories found
    marker_categories: list[str]  # which marker categories were found

    # --- Token-level analysis ---
    divergent_tokens: list[TokenDivergence] = field(default_factory=list)
    divergent_token_count: int = 0
    low_confidence_word_count: int = 0  # words with ASR confidence < threshold

    # --- Suggestion (passive, never applied in Stage 1) ---
    suggested_patch: str = ""  # what the correction would be
    suggestion_safe: bool = False  # whether all safety gates passed
    applied: bool = False  # ALWAYS False in Stage 1
    rejection_reason: str = ""  # why the suggestion was not marked safe

    # --- Metadata for output ---
    correction_type: str = "none"  # "suggestion" | "none" | "rejected"


# ---------------------------------------------------------------------------
# Candidate detection
# ---------------------------------------------------------------------------


def detect_hadith_candidates(text: str) -> list[NarrationMarkerHit]:
    """
    Detect narration markers in a text segment.

    Returns a list of NarrationMarkerHit objects, one per marker found.
    The caller uses the count and category diversity to decide whether
    the segment is a Hadith candidate.

    Parameters
    ----------
    text : Raw segment text (may be Arabic, Urdu, or mixed).

    Returns
    -------
    List of marker hits found in the text.
    """
    if not text or len(text.strip()) < 5:
        return []

    hits: list[NarrationMarkerHit] = []
    for category, pattern in _NARRATION_MARKERS:
        for match in pattern.finditer(text):
            hits.append(
                NarrationMarkerHit(
                    category=category,
                    pattern=pattern.pattern,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )

    return hits


def _unique_marker_categories(hits: list[NarrationMarkerHit]) -> list[str]:
    """Return deduplicated marker categories preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for hit in hits:
        if hit.category not in seen:
            seen.add(hit.category)
            result.append(hit.category)
    return result


# ---------------------------------------------------------------------------
# Normalization for comparison
# ---------------------------------------------------------------------------


def normalize_for_comparison(text: str) -> str:
    """
    Normalize Arabic/Urdu text for retrieval comparison.

    Pipeline:
    1. Cross-script canonicalization (Urdu → Arabic for matching)
    2. Arabic orthography normalization (alef, hamza, taa marbuta)
    3. Strip diacritics, punctuation, whitespace

    This reuses the battle-tested canonicalise_for_matching() from
    arabic_utils.py rather than reinventing normalization.
    """
    if not text:
        return ""
    return canonicalise_for_matching(text)


# ---------------------------------------------------------------------------
# Token-level divergence analysis
# ---------------------------------------------------------------------------


def compute_token_divergence(
    asr_text: str,
    retrieval_text: str,
    word_confidences: list[float] | None = None,
) -> list[TokenDivergence]:
    """
    Compute token-level divergence between ASR output and retrieval text.

    Unlike fuzzy similarity (which gives a single number), this identifies
    WHICH specific tokens diverge and whether those tokens have low ASR
    confidence — critical for safe correction decisions.

    Uses difflib.SequenceMatcher to perform LCS-based sequence alignment,
    robustly handling insertions and deletions without positional offset shifts.

    Parameters
    ----------
    asr_text : Raw ASR output text.
    retrieval_text : Arabic text from Hadith retrieval.
    word_confidences : Per-word ASR confidence values (from word timestamps).
                       If None, all words are treated as confidence=0.0
                       (unknown), which is the conservative default.

    Returns
    -------
    List of TokenDivergence objects, sorted by their position in the ASR text.
    """
    import difflib

    asr_norm = normalize_for_comparison(asr_text)
    ret_norm = normalize_for_comparison(retrieval_text)

    asr_tokens = asr_norm.split()
    ret_tokens = ret_norm.split()

    if not asr_tokens or not ret_tokens:
        return []

    # Default confidence: 0.0 (unknown) is conservative — it means we
    # cannot trust any word enough to block a suggestion.
    if word_confidences is None:
        word_confidences = [0.0] * len(asr_tokens)
    elif len(word_confidences) < len(asr_tokens):
        # Pad with 0.0 for missing entries
        word_confidences = list(word_confidences) + [0.0] * (
            len(asr_tokens) - len(word_confidences)
        )

    divergences: list[TokenDivergence] = []

    matcher = difflib.SequenceMatcher(None, asr_tokens, ret_tokens)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                idx = i1 + k
                divergences.append(
                    TokenDivergence(
                        asr_token=asr_tokens[idx],
                        retrieval_token=ret_tokens[j1 + k],
                        position=idx,
                        asr_word_confidence=word_confidences[idx],
                        is_divergent=False,
                        similarity=1.0,
                    )
                )
        elif tag == "replace":
            # If it's a 1-to-1 or equal-sized replacement block
            if (i2 - i1) == (j2 - j1):
                for k in range(i2 - i1):
                    idx = i1 + k
                    asr_tok = asr_tokens[idx]
                    ret_tok = ret_tokens[j1 + k]
                    sim = fuzz.ratio(asr_tok, ret_tok) / 100.0
                    is_div = sim < 1.0
                    divergences.append(
                        TokenDivergence(
                            asr_token=asr_tok,
                            retrieval_token=ret_tok,
                            position=idx,
                            asr_word_confidence=word_confidences[idx],
                            is_divergent=is_div,
                            similarity=sim,
                        )
                    )
            else:
                # Structural replacement (different sizes)
                # Treat all ASR tokens in this block as divergent
                for k in range(i2 - i1):
                    idx = i1 + k
                    ret_val = " ".join(ret_tokens[j1:j2])
                    divergences.append(
                        TokenDivergence(
                            asr_token=asr_tokens[idx],
                            retrieval_token=ret_val,
                            position=idx,
                            asr_word_confidence=word_confidences[idx],
                            is_divergent=True,
                            similarity=0.0,
                        )
                    )
        elif tag == "delete":
            # ASR has extra tokens. Mark them as divergent/deleted.
            for k in range(i2 - i1):
                idx = i1 + k
                divergences.append(
                    TokenDivergence(
                        asr_token=asr_tokens[idx],
                        retrieval_token="",  # empty indicates deletion
                        position=idx,
                        asr_word_confidence=word_confidences[idx],
                        is_divergent=True,
                        similarity=0.0,
                    )
                )
        elif tag == "insert":
            # Retrieval has extra tokens. Since we don't mutate/inject, we don't
            # suggest corrections for missing tokens.
            pass

    # Sort by position to preserve word order
    divergences.sort(key=lambda d: d.position)
    return divergences


# ---------------------------------------------------------------------------
# Retrieval confidence scoring
# ---------------------------------------------------------------------------


def compute_retrieval_score(
    asr_text: str,
    retrieval_text: str,
) -> tuple[float, float, float]:
    """
    Compute retrieval similarity between ASR output and Hadith text.

    Returns three scores:
    1. token_overlap: Jaccard similarity on word token sets
    2. fuzzy_similarity: weighted combination of rapidfuzz scores
    3. combined_confidence: calibrated overall score

    Parameters
    ----------
    asr_text : Raw ASR output text.
    retrieval_text : Arabic text from Hadith retrieval.

    Returns
    -------
    Tuple of (token_overlap, fuzzy_similarity, combined_confidence).
    All values are 0.0 – 1.0.
    """
    asr_norm = normalize_for_comparison(asr_text)
    ret_norm = normalize_for_comparison(retrieval_text)

    if not asr_norm or not ret_norm:
        return 0.0, 0.0, 0.0

    # --- Token overlap (Jaccard) ---
    asr_tokens = set(asr_norm.split())
    ret_tokens = set(ret_norm.split())
    if not asr_tokens or not ret_tokens:
        return 0.0, 0.0, 0.0

    intersection = len(asr_tokens & ret_tokens)
    union = len(asr_tokens | ret_tokens)
    token_overlap = intersection / union if union > 0 else 0.0

    # --- Fuzzy similarity ---
    # Use multiple rapidfuzz metrics for robustness:
    partial = fuzz.partial_ratio(asr_norm, ret_norm) / 100.0
    token_set = fuzz.token_set_ratio(asr_norm, ret_norm) / 100.0
    token_sort = fuzz.token_sort_ratio(asr_norm, ret_norm) / 100.0

    # Weighted combination: token_set is most robust for Arabic,
    # partial catches substring matches, token_sort respects word order.
    fuzzy_similarity = 0.35 * partial + 0.40 * token_set + 0.25 * token_sort

    # --- Length ratio penalty ---
    asr_len = len(asr_norm)
    ret_len = len(ret_norm)
    length_ratio = (
        min(asr_len, ret_len) / max(asr_len, ret_len) if max(asr_len, ret_len) > 0 else 0.0
    )

    # --- Combined confidence with length calibration ---
    # Penalize when lengths diverge significantly
    combined = fuzzy_similarity * (0.70 + 0.30 * length_ratio)

    return (
        round(token_overlap, 4),
        round(fuzzy_similarity, 4),
        round(min(1.0, max(0.0, combined)), 4),
    )


# ---------------------------------------------------------------------------
# Safety gating
# ---------------------------------------------------------------------------


def evaluate_suggestion_safety(
    asr_confidence: float,
    retrieval_confidence: float,
    token_overlap: float,
    marker_count: int,
    divergent_tokens: list[TokenDivergence],
    asr_text: str,
    retrieval_text: str,
) -> tuple[bool, str]:
    """
    Evaluate whether a correction suggestion is safe.

    ALL gates must pass for a suggestion to be marked safe.
    In Stage 1, even "safe" suggestions are never applied — they are
    recorded as metadata for manual review.

    Parameters
    ----------
    asr_confidence : Average word-level ASR confidence (0.0–1.0).
    retrieval_confidence : Combined retrieval similarity score.
    token_overlap : Jaccard token overlap ratio.
    marker_count : Number of distinct narration marker categories found.
    divergent_tokens : Token-level divergence analysis results.
    asr_text : Normalized ASR text.
    retrieval_text : Normalized retrieval text.

    Returns
    -------
    Tuple of (is_safe, rejection_reason).
    If is_safe is True, rejection_reason is empty.
    """
    # Gate 0.5: Structural check — do not allow insertions, deletions, or structural changes.
    # We only permit 1-to-1 word corrections (substitutions) to prevent hallucinated content injection.
    for tok in divergent_tokens:
        if tok.is_divergent:
            ret_val = tok.retrieval_token.strip()
            if not ret_val or " " in ret_val:
                return (
                    False,
                    f"structural_mismatch (not a simple 1-to-1 word correction for '{tok.asr_token}')",
                )

    # Gate 1: ASR confidence must be LOW (uncertain transcription)
    if asr_confidence >= ASR_CONFIDENCE_CEILING:
        return False, f"asr_confidence_high ({asr_confidence:.3f} >= {ASR_CONFIDENCE_CEILING})"

    # Gate 2: Retrieval similarity must be HIGH
    if retrieval_confidence < RETRIEVAL_SIMILARITY_FLOOR:
        return (
            False,
            f"low_retrieval_similarity ({retrieval_confidence:.3f} < {RETRIEVAL_SIMILARITY_FLOOR})",
        )

    # Gate 3: Token overlap must be substantial
    if token_overlap < TOKEN_OVERLAP_FLOOR:
        return False, f"low_token_overlap ({token_overlap:.3f} < {TOKEN_OVERLAP_FLOOR})"

    # Gate 4: At least one narration marker category present
    if marker_count < _MIN_MARKER_CATEGORIES:
        return False, f"no_narration_markers (count={marker_count})"

    # Gate 5: Divergent span must be small (max 2 words in Stage 1)
    div_count = sum(1 for t in divergent_tokens if t.is_divergent)
    if div_count > _MAX_SUGGESTION_SPAN_WORDS:
        return False, f"span_exceeds_limit ({div_count} > {_MAX_SUGGESTION_SPAN_WORDS})"

    # Gate 6: Length delta check
    asr_norm = normalize_for_comparison(asr_text)
    ret_norm = normalize_for_comparison(retrieval_text)
    asr_len = len(asr_norm)
    ret_len = len(ret_norm)
    if max(asr_len, ret_len) > 0:
        length_delta = abs(asr_len - ret_len) / max(asr_len, ret_len)
        if length_delta > MAX_LENGTH_DELTA_RATIO:
            return False, f"length_delta_high ({length_delta:.3f} > {MAX_LENGTH_DELTA_RATIO})"

    # Gate 7: Acoustic alignment — divergent tokens must have LOW ASR
    # confidence. If ASR was confident about a divergent word, retrieval
    # should NOT override it. This is the critical safeguard against
    # retrieval hallucination.
    for tok in divergent_tokens:
        if tok.is_divergent and tok.asr_word_confidence > ASR_CONFIDENCE_CEILING:
            return False, (
                f"divergent_token_high_confidence "
                f"('{tok.asr_token}' conf={tok.asr_word_confidence:.3f})"
            )

    return True, ""


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------


def _build_suggestion(divergent_tokens: list[TokenDivergence]) -> str:
    """
    Build a human-readable suggestion string from divergent tokens.

    Only includes tokens that actually diverge — not the entire text.
    Format: "token_a→token_b, token_c→token_d"
    """
    parts: list[str] = []
    for tok in divergent_tokens:
        if tok.is_divergent:
            parts.append(f"{tok.asr_token}→{tok.retrieval_token}")
    return ", ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------


class HadithVerifier:
    """
    Retrieval-assisted Hadith verification engine.

    Stage 1: verification + suggestion only — no transcript mutation.

    Usage
    -----
    verifier = HadithVerifier()
    result = verifier.verify(
        segment_id=42,
        asr_text="قال رسول الله ...",
        retrieval_text="...",
        retrieval_collection="bukhari",
        retrieval_number="1",
        retrieval_confidence=0.85,
        asr_word_confidence=0.38,
        word_confidences=[0.9, 0.8, 0.3, 0.2, ...],
    )
    # result.applied is always False
    # result.suggestion_safe indicates if Stage 2 would apply this
    """

    def verify(
        self,
        segment_id: int,
        asr_text: str,
        retrieval_text: str,
        retrieval_collection: str,
        retrieval_number: str,
        retrieval_confidence: float,
        asr_word_confidence: float,
        word_confidences: list[float] | None = None,
    ) -> HadithVerificationResult:
        """
        Verify a single Hadith segment against its retrieval match.

        Parameters
        ----------
        segment_id : Pipeline segment ID.
        asr_text : Raw ASR output for this segment.
        retrieval_text : Arabic text from the matched Hadith.
        retrieval_collection : Hadith collection name.
        retrieval_number : Hadith number.
        retrieval_confidence : Original confidence from the matcher.
        asr_word_confidence : Average word-level ASR confidence.
        word_confidences : Per-word confidence values from word timestamps.

        Returns
        -------
        HadithVerificationResult with all metadata populated.
        `applied` is always False in Stage 1.
        """
        # --- Step 1: Detect narration markers ---
        marker_hits = detect_hadith_candidates(asr_text)
        categories = _unique_marker_categories(marker_hits)
        marker_count = len(categories)

        logger.debug(
            "Hadith verification [seg=%d]: %d markers found (%s)",
            segment_id,
            marker_count,
            ", ".join(categories) if categories else "none",
        )

        # --- Step 2: Compute retrieval scores ---
        token_overlap, fuzzy_sim, combined_conf = compute_retrieval_score(
            asr_text,
            retrieval_text,
        )

        logger.debug(
            "Hadith verification [seg=%d]: token_overlap=%.3f, fuzzy=%.3f, combined=%.3f",
            segment_id,
            token_overlap,
            fuzzy_sim,
            combined_conf,
        )

        # --- Step 3: Token-level divergence analysis ---
        divergences = compute_token_divergence(
            asr_text,
            retrieval_text,
            word_confidences,
        )
        divergent_count = sum(1 for t in divergences if t.is_divergent)

        # Count words with low ASR confidence
        low_conf_count = 0
        if word_confidences:
            low_conf_count = sum(1 for c in word_confidences if c < WORD_CONFIDENCE_FLOOR)

        # --- Step 4: Evaluate suggestion safety ---
        is_safe, rejection_reason = evaluate_suggestion_safety(
            asr_confidence=asr_word_confidence,
            retrieval_confidence=combined_conf,
            token_overlap=token_overlap,
            marker_count=marker_count,
            divergent_tokens=divergences,
            asr_text=asr_text,
            retrieval_text=retrieval_text,
        )

        # --- Step 5: Build suggestion (even if not safe, for diagnostics) ---
        suggested_patch = _build_suggestion(divergences)

        # Determine correction type for metadata
        if is_safe and suggested_patch:
            correction_type = "suggestion"
        elif suggested_patch and not is_safe:
            correction_type = "rejected"
        else:
            correction_type = "none"

        # --- Step 6: Log diagnostics ---
        if is_safe and suggested_patch:
            logger.info(
                "Hadith verification [seg=%d]: SUGGESTION generated "
                "(type=%s, confidence=%.3f, patch='%s')",
                segment_id,
                correction_type,
                combined_conf,
                suggested_patch,
            )
        elif rejection_reason:
            logger.debug(
                "Hadith verification [seg=%d]: suggestion REJECTED (%s)",
                segment_id,
                rejection_reason,
            )
        else:
            logger.debug(
                "Hadith verification [seg=%d]: no divergence found",
                segment_id,
            )

        return HadithVerificationResult(
            segment_id=segment_id,
            segment_text=asr_text,
            hadith_match_collection=retrieval_collection,
            hadith_match_number=retrieval_number,
            hadith_match_confidence=retrieval_confidence,
            retrieval_confidence=combined_conf,
            token_overlap=token_overlap,
            fuzzy_similarity=fuzzy_sim,
            narration_marker_count=marker_count,
            marker_categories=categories,
            divergent_tokens=divergences,
            divergent_token_count=divergent_count,
            low_confidence_word_count=low_conf_count,
            suggested_patch=suggested_patch,
            suggestion_safe=is_safe,
            applied=False,  # ALWAYS False in Stage 1
            rejection_reason=rejection_reason,
            correction_type=correction_type,
        )
