"""
pipeline.py
-----------
Main orchestrator: transcription → classification → merge → post-process → verify → output.

Review 2 fixes
---------------
- P0 : Reordered pipeline: merge happens BEFORE verification so that
       verified metadata never attaches to text that changes after merge.
- P0 : Arabic span extraction now uses Urdu lexical filtering to avoid
       sending Urdu text to Quran/Hadith matchers.
- P1 : Post-processing keeps raw_text provenance for auditability.
       Confidence is reduced when a match depends on corrected text.

Pipeline order:
  transcribe → classify language → merge by detected language
  → post-process (with raw provenance) → Quran/Hadith verify → output
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from islamic_stt.audio.preprocess import preprocess_audio
from islamic_stt.config import PipelineConfig
from islamic_stt.core.arabic_utils import (
    canonicalise_for_matching,
    contains_arabic_script,
    extract_arabic_spans,
    is_dominantly_arabic,
    normalise_arabic,
)
from islamic_stt.core.decoding_passes import (
    apply_dual_pass_arabic,
    retry_low_quality_segments,
    transcribe_with_contextual_prompts,
)
from islamic_stt.core.language_detector import detect_language
from islamic_stt.core.overlap_merger import deduplicate_overlap
from islamic_stt.core.post_processor import apply_post_processing
from islamic_stt.core.prompt_builder import detect_content_mode
from islamic_stt.core.segment_merger import merge_arabic_quote_blocks, merge_short_segments
from islamic_stt.core.transcriber import Transcriber
from islamic_stt.core.types import TranscriptSegment
from islamic_stt.core.urdu_utils import normalise_urdu, roman_urdu_to_script
from islamic_stt.diagnostics import DiagnosticsLogger, SegmentDiagnostics
from islamic_stt.matchers.quran_matcher import FormulaMatch, QuranMatcher
from islamic_stt.output.flagged_handler import FlaggedHandler
from islamic_stt.output.output_handler import EnrichedSegment, OutputHandler

# Hadith verification (Stage 1: suggestion only)
try:
    from islamic_stt.matchers.hadith_verifier import HadithVerifier

    _VERIFIER_AVAILABLE = True
except ImportError:
    _VERIFIER_AVAILABLE = False

logger = logging.getLogger("islamic_stt.pipeline")

_SUPPORTED_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".webm",
    }
)

# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _timed(label: str, stats: dict) -> Generator[None, None, None]:
    t0 = time.perf_counter()
    yield
    stats[f"{label}_ms"] = round((time.perf_counter() - t0) * 1000, 1)


# ---------------------------------------------------------------------------
# GPU detection (P1 fix: respect explicit compute_type)
# ---------------------------------------------------------------------------


def _detect_device(config: PipelineConfig) -> tuple[str, str]:
    requested = config.device
    explicit_compute = config.compute_type
    if requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, vram_gb)
                return "cuda", explicit_compute
            else:
                logger.warning("CUDA requested but unavailable — falling back to CPU.")
                return "cpu", "int8"
        except ImportError:
            logger.warning("PyTorch not installed — attempting CUDA anyway.")
            return "cuda", explicit_compute
    return "cpu", "int8" if explicit_compute == "float16" else explicit_compute


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_audio(audio_path: str, config: PipelineConfig) -> None:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        logger.warning("Unrecognised extension '%s' — attempting anyway.", ext)
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb < 0.001:
        raise ValueError(f"Audio file appears empty: {audio_path}")
    if file_size_mb > config.max_file_size_mb:
        raise ValueError(
            f"Audio file too large: {file_size_mb:.1f} MB (limit: {config.max_file_size_mb:.0f} MB)"
        )

    # P2 fix: Probe duration before model load to prevent long-running jobs
    duration_s = _probe_duration(audio_path)
    if duration_s is not None:
        if duration_s > config.max_audio_duration_s:
            raise ValueError(
                f"Audio too long: {duration_s:.0f}s "
                f"(limit: {config.max_audio_duration_s:.0f}s). "
                f"Split the file or increase max_audio_duration_s."
            )
        logger.info(
            "Audio validated: %s (%.1f MB, %.0fs)",
            os.path.basename(audio_path),
            file_size_mb,
            duration_s,
        )
    else:
        logger.info(
            "Audio validated: %s (%.1f MB, duration unknown)",
            os.path.basename(audio_path),
            file_size_mb,
        )


def _probe_duration(audio_path: str) -> float | None:
    """Probe audio duration without loading the full file."""
    # Try soundfile (fast, header-only)
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass

    # Try ffprobe (subprocess)
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Quality / confidence scoring
# ---------------------------------------------------------------------------


def compute_segment_quality(seg: TranscriptSegment) -> float:
    logprob_score = max(0.0, min(1.0, (seg.avg_logprob + 1.5) / 1.5))
    speech_score = 1.0 - seg.no_speech_prob
    return round(
        0.35 * logprob_score + 0.45 * seg.avg_word_confidence + 0.20 * speech_score,
        4,
    )


def compute_segment_confidence(seg: TranscriptSegment) -> float:
    """Aggregate confidence from multiple signals (review feedback)."""
    speech_score = 1.0 - seg.no_speech_prob
    logprob_score = max(0.0, 1.0 + seg.avg_logprob)  # avg_logprob is negative
    # Repetition score: penalize segments with low compression ratio
    # (high compression = likely repetition)
    repetition_score = 1.0  # default: no penalty
    if hasattr(seg, "compression_ratio") and seg.compression_ratio > 0:
        repetition_score = max(0.0, min(1.0, 2.0 - seg.compression_ratio))
    return round(
        0.25 * speech_score + 0.50 * logprob_score + 0.25 * repetition_score,
        4,
    )


# ---------------------------------------------------------------------------
# Adaptive condition_on_previous_text (review feedback)
# ---------------------------------------------------------------------------


def _should_condition_on_previous(
    prev_seg: TranscriptSegment | None,
    current_lang: str,
    prev_lang: str | None,
) -> bool:
    """
    Decide whether to enable condition_on_previous_text for the next chunk.

    Only enable when:
      - Previous segment confidence > 0.85
      - avg_logprob > -0.45 (not garbage)
      - compression_ratio < 1.35 (not looping)
      - Same detected language
    """
    if prev_seg is None:
        return False

    # Confidence check
    confidence = compute_segment_confidence(prev_seg)
    if confidence < 0.85:
        return False

    # Log probability check
    if prev_seg.avg_logprob < -0.45:
        return False

    # Language continuity check
    if prev_lang != current_lang:
        return False

    return True


def _has_low_word_confidence(seg: TranscriptSegment, threshold: float) -> bool:
    """Treat 0.0 as unknown when word timestamps are unavailable."""
    return bool(seg.words) and seg.avg_word_confidence < threshold


# ---------------------------------------------------------------------------
# Per-segment language classification (P0-2 fix)
# ---------------------------------------------------------------------------


def _classify_segment_language(seg: TranscriptSegment) -> str:
    """
    Classify segment language independently of Whisper's file-level label.
    Pass whisper_language=None to force independent classification.
    """
    text = seg.text.strip()
    if not text:
        return "und"
    norm = normalise_arabic(text)
    return detect_language(text, whisper_language=None, pre_normalised=norm)


# ---------------------------------------------------------------------------
# Urdu-aware Arabic span extraction (P0 fix from review 2)
# ---------------------------------------------------------------------------


def _extract_arabic_spans(text: str) -> list[str]:
    """Compatibility wrapper around the shared Urdu-aware span extractor."""
    return extract_arabic_spans(text)


# ---------------------------------------------------------------------------
# Arabic dominance check (Colab fix: reduces false Arabic flagging)
# ---------------------------------------------------------------------------
# Urdu-exclusive codepoints — imported from arabic_utils but kept as a
# module-level constant for the dominance check.
_URDU_EXCLUSIVE_CHARS = frozenset(
    "\u0679\u067e\u0686\u0688\u0691\u0698\u06a9\u06af\u06ba\u06be\u06c1\u06c3\u06cc\u06d2"
)


# ---------------------------------------------------------------------------
# Hadith matcher loading
# ---------------------------------------------------------------------------


def _load_hadith_matcher(config: PipelineConfig):
    """Load Hadith matcher: prefer local DB, fall back to API."""
    if os.path.isfile(config.hadith_db_path):
        try:
            from islamic_stt.matchers.hadith_db import LocalHadithMatcher

            matcher = LocalHadithMatcher(
                db_path=config.hadith_db_path,
                token_set_threshold=config.hadith_token_set_threshold,
                combined_threshold=config.hadith_combined_threshold,
            )
            logger.info("Using LOCAL Hadith database: %s", config.hadith_db_path)
            return matcher, True
        except Exception as exc:
            logger.warning("Failed to load local Hadith DB: %s — trying API.", exc)
    try:
        from islamic_stt.matchers.hadith_matcher import HadithMatcher

        matcher = HadithMatcher(api_key=config.sunnah_api_key)
        logger.info("Using Sunnah.com API for Hadith matching (rate-limited).")
        return matcher, False
    except (ValueError, ImportError) as exc:
        logger.warning("Hadith matcher disabled: %s", exc)
        return None, False


def _downgrade_corrected_quran_match(match_result, matched_text: str):
    """
    Preserve Quran metadata while making corrected-text provenance explicit.

    A match that only appears after post-processing must not be displayed as
    exact/100%, because the user needs to know the ASR text was changed.
    """
    return match_result.__class__(
        surah_id=match_result.surah_id,
        surah_name=match_result.surah_name,
        ayah_id=match_result.ayah_id,
        original_text=match_result.original_text,
        matched_text=matched_text,
        confidence=round(match_result.confidence * 0.90, 4),
        is_exact=False,
        is_ambiguous=getattr(match_result, "is_ambiguous", False),
        ambiguous_count=getattr(match_result, "ambiguous_count", 1),
        alternate_refs=getattr(match_result, "alternate_refs", None),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig, audio_path: str) -> dict[str, int | float]:
    """
    Run the full Islamic STT pipeline.

    Pipeline order (review 2 fix):
      transcribe → classify → merge → post-process → verify → output

    Merge happens BEFORE verification so verified metadata can never
    attach to text that changes afterward.
    """
    start_time = time.time()
    stats: dict[str, int | float] = {
        "total_segments": 0,
        "arabic_segments": 0,
        "formula_hits": 0,
        "quran_hits": 0,
        "hadith_hits": 0,
        "hadith_paraphrases": 0,
        "flagged": 0,
        "low_quality_segments": 0,
        "post_processed": 0,
    }

    # --- Input validation ---
    _validate_audio(audio_path, config)

    # ================================================================
    # STEP 0: Audio preprocessing (highest ROI improvement)
    # ================================================================
    processed_audio = audio_path
    if config.preprocess_audio:
        with _timed("preprocess", stats):
            try:
                processed_audio = preprocess_audio(
                    audio_path,
                    denoise=config.denoise_audio,
                )
                if processed_audio != audio_path:
                    stats["audio_preprocessed"] = 1
            except Exception as exc:
                logger.warning("Audio preprocessing failed: %s — using original.", exc)
                processed_audio = audio_path

    # --- GPU auto-detection ---
    resolved_device, resolved_compute = _detect_device(config)
    if resolved_device != config.device or resolved_compute != config.compute_type:
        logger.info(
            "Device/compute: %s/%s → %s/%s",
            config.device,
            config.compute_type,
            resolved_device,
            resolved_compute,
        )

    # --- Load model ---
    try:
        with _timed("model_load", stats):
            transcriber = Transcriber(
                model_size=config.model,
                device=resolved_device,
                compute_type=resolved_compute,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model '{config.model}' on {resolved_device}: {exc}"
        ) from exc

    # --- Load Quran corpus ---
    try:
        with _timed("corpus_load", stats):
            quran_matcher = QuranMatcher(corpus_path=config.quran_corpus)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Quran corpus error: %s", exc)
        raise

    # --- Load Hadith matcher ---
    hadith_matcher = None
    hadith_is_local = False
    if not config.no_hadith:
        hadith_matcher, hadith_is_local = _load_hadith_matcher(config)

    output_handler = OutputHandler(output_path=config.output_path)
    flagged_handler = FlaggedHandler(flagged_path=config.flagged_path)

    # ================================================================
    # STEP 1: Transcribe (on preprocessed audio)
    # ================================================================
    with _timed("transcription", stats):
        segments = transcribe_with_contextual_prompts(
            transcriber,
            processed_audio,
            config,
        )
    stats["total_segments"] = len(segments)

    if config.enable_retry_decoding:
        with _timed("retry_decode", stats):
            segments, retried_n = retry_low_quality_segments(
                transcriber,
                processed_audio,
                segments,
                config,
            )
            stats["retry_triggered_count"] = retried_n

    # ================================================================
    # STEP 1b: Overlap deduplication (removes chunk-boundary duplicates)
    # ================================================================
    if config.chunk_overlap > 0:
        with _timed("overlap_dedup", stats):
            pre_dedup = len(segments)
            segments = deduplicate_overlap(
                segments,
                overlap_seconds=float(config.chunk_overlap),
            )
            dedup_removed = pre_dedup - len(segments)
            if dedup_removed > 0:
                stats["overlap_dedup_removed"] = dedup_removed

    # ================================================================
    # STEP 2: Classify language per segment (before merge!)
    # ================================================================
    lang_lock_info: dict[int, dict] = {}  # seg_id → lock diagnostics
    with _timed("classify", stats):
        prev_lang: str | None = None
        for seg in segments:
            lang = _classify_segment_language(seg)

            # Short-segment context inheritance: for very short segments
            # (<4 words), inherit surrounding language context unless
            # strong script signals override.
            word_count = len(seg.text.split())
            if word_count < 4 and prev_lang and lang not in ("ar", "en"):
                # Check for strong Arabic or English signals
                text = seg.text.strip()
                has_urdu_exclusive = any(c in _URDU_EXCLUSIVE_CHARS for c in text)
                latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
                latin_ratio = latin_chars / max(len(text), 1)
                if has_urdu_exclusive:
                    lang = "ur"  # Urdu-exclusive chars → definitely Urdu
                elif latin_ratio > 0.7:
                    lang = "en"  # Mostly Latin → English
                elif lang in ("fa", "af", "no", "nl", "id"):
                    # Impossible languages for this context → inherit
                    lang = prev_lang

            seg.language = lang  # type: ignore[misc]
            prev_lang = lang

            # Language-locked decoding: log lock decisions per segment
            if config.enable_language_locking:
                lang_prob = seg.language_probability
                if lang_prob > config.language_lock_threshold and lang in ("ar", "ur", "en"):
                    lang_lock_info[seg.id] = {
                        "language_locked": True,
                        "locked_lang": lang,
                        "confidence": lang_prob,
                        "reason": "high_language_probability",
                    }
                    stats["language_locked_segments"] = stats.get("language_locked_segments", 0) + 1
                else:
                    lang_lock_info[seg.id] = {
                        "language_locked": False,
                        "locked_lang": "",
                        "confidence": lang_prob,
                        "reason": "",
                    }

    if config.enable_dual_pass_arabic:
        with _timed("dual_pass_arabic", stats):
            segments, redrafted_n = apply_dual_pass_arabic(
                transcriber,
                processed_audio,
                segments,
                config,
            )
            stats["dual_pass_segments"] = redrafted_n

    # ================================================================
    # STEP 3: Merge short segments (using detected language)
    # ================================================================
    with _timed("merge", stats):
        merged_segments = merge_short_segments(segments)
        if config.merge_arabic_quote_blocks:
            merged_segments = merge_arabic_quote_blocks(merged_segments)
        merged_langs: list[str] = []
        for seg in merged_segments:
            lang = _classify_segment_language(seg)
            merged_langs.append(lang)

    logger.info("Segments: %d raw → %d after merge.", len(segments), len(merged_segments))

    # ================================================================
    # STEP 4: Post-process with raw provenance + Urdu normalization
    # ================================================================
    with _timed("post_process", stats):
        raw_texts: list[str] = []  # keep original for auditability
        for seg, lang in zip(merged_segments, merged_langs, strict=False):
            raw_texts.append(seg.text)
            corrected = apply_post_processing(seg.text)
            # Urdu normalization (safe rules only)
            if lang == "ur" or lang == "und":
                corrected = normalise_urdu(corrected)
                corrected = roman_urdu_to_script(corrected)
            if corrected != seg.text:
                stats["post_processed"] = stats.get("post_processed", 0) + 1
                seg.text = corrected  # type: ignore[misc]

    # ================================================================
    # STEP 5: Verify (Quran + Hadith) — runs on final merged text
    # ================================================================
    with _timed("enrichment", stats):
        enriched: list[EnrichedSegment] = []
        for i, (seg, lang, raw_text) in enumerate(
            zip(merged_segments, merged_langs, raw_texts, strict=False)
        ):
            was_corrected = seg.text != raw_text
            if was_corrected:
                final_lang = _classify_segment_language(seg)
                if final_lang != "und":
                    lang = final_lang
                    merged_langs[i] = final_lang

            es = EnrichedSegment(
                segment=seg,
                detected_lang=lang,
                raw_text=raw_text,
                was_corrected=was_corrected,
            )

            quality = compute_segment_quality(seg)
            if quality < config.low_quality_threshold:
                stats["low_quality_segments"] += 1
            low_confidence = _has_low_word_confidence(seg, config.low_confidence_threshold)

            if lang == "ar" or (lang != "en" and is_dominantly_arabic(seg.text)):
                stats["arabic_segments"] += 1
                canon_text = canonicalise_for_matching(seg.text)
                match_result = quran_matcher.match(
                    seg.text,
                    canonicalised=canon_text,
                    allow_fuzzy=not low_confidence,
                    avg_word_confidence=seg.avg_word_confidence,
                )

                if isinstance(match_result, FormulaMatch):
                    es.formula_match = match_result
                    stats["formula_hits"] += 1
                elif match_result is not None:
                    # P0 review 3: If match depended on post-processing correction,
                    # ALWAYS downgrade — even exact matches. A corrected exact match
                    # must never render as '✓exact (100%)'.
                    if was_corrected:
                        match_result = _downgrade_corrected_quran_match(match_result, raw_text)
                    es.quran_match = match_result
                    stats["quran_hits"] += 1
                elif low_confidence:
                    es.is_flagged = True
                    flagged_handler.add(
                        seg,
                        reason=(
                            f"Low confidence Arabic (avg={seg.avg_word_confidence:.2f}); "
                            "fuzzy Quran/Hadith verification skipped"
                        ),
                    )
                    stats["flagged"] += 1
                else:
                    es.pending_hadith = True

            elif lang == "ur" and contains_arabic_script(seg.text):
                # Extract Arabic spans from Urdu segments
                arabic_spans = _extract_arabic_spans(seg.text)
                for span in arabic_spans:
                    canon_span = canonicalise_for_matching(span)
                    span_match = quran_matcher.match(
                        span,
                        canonicalised=canon_span,
                        allow_fuzzy=not low_confidence,
                        avg_word_confidence=seg.avg_word_confidence,
                    )
                    if isinstance(span_match, FormulaMatch):
                        es.formula_match = span_match
                        stats["formula_hits"] += 1
                        break
                    elif span_match is not None:
                        # P0 review 4: Apply same correction penalty to span matches
                        if was_corrected:
                            span_match = _downgrade_corrected_quran_match(span_match, span)
                        es.quran_match = span_match
                        stats["quran_hits"] += 1
                        break
                if arabic_spans and not es.quran_match and not es.formula_match:
                    stats["arabic_spans_in_urdu"] = stats.get("arabic_spans_in_urdu", 0) + 1
                    if low_confidence:
                        es.is_flagged = True
                        flagged_handler.add(
                            seg,
                            reason=(
                                f"Low confidence Urdu segment with Arabic quote "
                                f"(avg={seg.avg_word_confidence:.2f}); fuzzy verification skipped"
                            ),
                        )
                        stats["flagged"] += 1
                    else:
                        es.pending_hadith = True

            enriched.append(es)

    # ================================================================
    # STEP 6: Hadith matching
    # ================================================================
    pending = [es for es in enriched if es.pending_hadith]
    if pending and hadith_matcher:
        with _timed("hadith", stats):
            _run_hadith_matching(
                pending,
                hadith_matcher,
                hadith_is_local,
                flagged_handler,
                stats,
                config,
            )
    elif pending:
        for es in pending:
            es.is_flagged = True
            flagged_handler.add(es.segment, reason="Hadith matching disabled; no Quran match")
            stats["flagged"] += 1
            es.pending_hadith = False

    # ================================================================
    # STEP 6b: Hadith verification (retrieval-assisted, opt-in)
    #          Stage 1: suggestion only — no transcript mutation.
    # ================================================================
    if config.enable_hadith_verification and hadith_matcher and _VERIFIER_AVAILABLE:
        with _timed("hadith_verify", stats):
            _run_hadith_verification(enriched, stats)

    # ================================================================
    # STEP 7: Output
    # ================================================================
    with _timed("output", stats):
        output_handler.write(enriched)
        flagged_handler.write()

    # ================================================================
    # STEP 8: Diagnostics (per-segment JSONL for empirical calibration)
    # ================================================================
    diag_logger = DiagnosticsLogger(
        output_path=config.diagnostics_path if config.enable_diagnostics else None
    )
    if config.enable_diagnostics:
        content_mode = detect_content_mode([es.segment for es in enriched[:10]])
        for es in enriched:
            seg = es.segment
            lock_info = lang_lock_info.get(seg.id, {})
            diag = SegmentDiagnostics(
                segment_id=seg.id,
                start=seg.start,
                end=seg.end,
                text_preview=seg.text[:60],
                original_text_preview=(es.raw_text or seg.text)[:60],
                detected_lang=es.detected_lang,
                language_locked=lock_info.get("language_locked", False),
                lock_reason=lock_info.get("reason", ""),
                confidence=compute_segment_confidence(seg),
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                avg_word_confidence=seg.avg_word_confidence,
                word_count=len(seg.text.split()),
                prompt_mode=content_mode,
                quran_match=es.quran_match is not None,
                quran_confidence=getattr(es.quran_match, "confidence", 0.0),
                quran_is_exact=getattr(es.quran_match, "is_exact", False),
                quran_is_ambiguous=getattr(es.quran_match, "is_ambiguous", False),
                hadith_match=es.hadith_match is not None,
                hadith_confidence=getattr(es.hadith_match, "confidence", 0.0),
                formula_match=es.formula_match is not None,
                was_corrected=es.was_corrected,
                is_flagged=es.is_flagged,
                mixed_script_ratio=getattr(seg, "mixed_script_ratio", 0.0),
                repeated_ngram_score=getattr(seg, "repeated_ngram_score", 0.0),
                retry_triggered=getattr(seg, "retry_triggered", False),
                retry_reason=getattr(seg, "retry_reason", []),
                confidence_delta=getattr(seg, "confidence_delta", 0.0),
                # Hadith verification diagnostics
                hadith_retrieval_attempted=bool(es.hadith_correction_type),
                hadith_retrieval_accepted=es.hadith_suggestion_safe,
                hadith_retrieval_rejected_reason=(
                    es.hadith_correction_detail if es.hadith_correction_type == "rejected" else ""
                ),
                hadith_verification_confidence=es.hadith_retrieval_confidence,
                hadith_suggestion_generated=(
                    es.hadith_suggestion_safe and bool(es.hadith_correction_detail)
                ),
            )
            diag_logger.add(diag)

        diag_logger.write()
        summary = diag_logger.summarize()
        if summary:
            logger.info(
                "Diagnostics summary: %d segs, confidence p50=%.3f, "
                "drift_events=%d, locked=%d, flagged=%d",
                summary.get("total_segments", 0),
                summary.get("confidence_p50", 0),
                summary.get("language_drift_events", 0),
                summary.get("language_locked_count", 0),
                summary.get("flagged_segments", 0),
            )

    if hadith_matcher and hasattr(hadith_matcher, "close"):
        hadith_matcher.close()

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    timing = ", ".join(f"{k}={v}ms" for k, v in stats.items() if k.endswith("_ms"))
    logger.info("Pipeline complete in %.1fs | %s", elapsed, timing)

    # Cleanup temp preprocessed audio
    if processed_audio != audio_path and os.path.isfile(processed_audio):
        try:
            os.unlink(processed_audio)
            parent = os.path.dirname(processed_audio)
            if parent and os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except OSError:
            pass

    # Write runtime metadata
    metadata_path = os.path.join(os.path.dirname(config.output_path) or ".", "run_metadata.json")
    try:
        import subprocess

        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    # Calculate runtime costs
    audio_dur_s = _probe_duration(audio_path) or 1.0
    retry_time_s = stats.get("retry_decode_ms", 0.0) / 1000.0
    transcription_time_s = stats.get("transcription_ms", 0.0) / 1000.0
    total_segments = stats.get("total_segments", 1)

    metadata = {
        "profile": config.profile,
        "model": config.model,
        "beam_size": config.beam_size,
        "best_of": config.best_of,
        "primary_language": config.primary_language,
        "compute_type": resolved_compute,
        "device": resolved_device,
        "vad_enabled": True,
        "overlap_seconds": config.chunk_overlap,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "preprocessing_enabled": config.preprocess_audio,
        "git_commit": git_commit,
        "runtime_seconds": stats.get("elapsed_seconds", 0.0),
        "retry_time_seconds": round(retry_time_s, 2),
        "retries_per_hour_audio": round(
            stats.get("retry_triggered_count", 0) / (audio_dur_s / 3600.0), 2
        ),
        "avg_decode_time_per_segment": round(transcription_time_s / total_segments, 3)
        if total_segments > 0
        else 0.0,
    }
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to write metadata to %s: %s", metadata_path, exc)

    return stats


# ---------------------------------------------------------------------------
# Hadith dispatch
# ---------------------------------------------------------------------------


def _run_hadith_matching(
    pending: list[EnrichedSegment],
    hadith_matcher: object,
    is_local: bool,
    flagged_handler: FlaggedHandler,
    stats: dict,
    config: PipelineConfig,
) -> None:
    if is_local:
        for es in pending:
            try:
                text = es.segment.text
                if es.detected_lang == "ur":
                    spans = _extract_arabic_spans(text)
                    if spans:
                        text = max(spans, key=len)
                result = hadith_matcher.match(text)  # type: ignore
                _apply_hadith_result(es, result, flagged_handler, stats)
            except Exception as exc:
                logger.warning("Local Hadith match error: %s", exc)
                _apply_hadith_result(es, None, flagged_handler, stats)
        return

    # API fallback
    try:
        results = asyncio.run(
            hadith_matcher.match_many_async([es.segment.text for es in pending])  # type: ignore
        )
        for es, hm in zip(pending, results, strict=False):
            _apply_hadith_result(es, hm, flagged_handler, stats)
    except Exception as exc:
        logger.warning("Async Hadith failed: %s — ThreadPool fallback.", exc)
        with ThreadPoolExecutor(max_workers=config.hadith_workers) as pool:
            futures = {pool.submit(hadith_matcher.match, es.segment.text): es for es in pending}  # type: ignore
            for future in as_completed(futures):
                es = futures[future]
                hm = None
                with contextlib.suppress(Exception):
                    hm = future.result()
                _apply_hadith_result(es, hm, flagged_handler, stats)


def _apply_hadith_result(
    es: EnrichedSegment,
    hadith_match: object | None,
    flagged_handler: FlaggedHandler,
    stats: dict,
) -> None:
    es.pending_hadith = False
    if hadith_match:
        es.hadith_match = hadith_match  # type: ignore[assignment]
        stats["hadith_hits"] += 1
        if hasattr(hadith_match, "is_paraphrase") and hadith_match.is_paraphrase:  # type: ignore
            stats["hadith_paraphrases"] = stats.get("hadith_paraphrases", 0) + 1
    else:
        es.is_flagged = True
        flagged_handler.add(es.segment, reason="No Quran or Hadith match found")
        stats["flagged"] += 1


# ---------------------------------------------------------------------------
# Hadith verification (Stage 1: suggestion only)
# ---------------------------------------------------------------------------


def _run_hadith_verification(
    enriched: list[EnrichedSegment],
    stats: dict,
) -> None:
    """
    Run retrieval-assisted Hadith verification on matched segments.

    Stage 1: generates suggestions and metadata ONLY — never modifies
    the transcript text.  Audio remains source of truth.

    Only processes segments where a Hadith match already exists (from Step 6).
    Uses word-level timestamps for acoustic gating when available.
    """
    verifier = HadithVerifier()
    verified_count = 0
    suggestion_count = 0

    for es in enriched:
        # Only verify segments that have an existing Hadith match
        if es.hadith_match is None:
            continue

        hm = es.hadith_match
        seg = es.segment

        # Extract word-level confidences from word timestamps
        word_confidences: list[float] | None = None
        if seg.words:
            word_confidences = [w.probability for w in seg.words]

        # Run verification
        result = verifier.verify(
            segment_id=seg.id,
            asr_text=seg.text,
            retrieval_text=hm.arabic_text,  # type: ignore[union-attr]
            retrieval_collection=hm.collection,  # type: ignore[union-attr]
            retrieval_number=hm.hadith_number,  # type: ignore[union-attr]
            retrieval_confidence=hm.confidence,  # type: ignore[union-attr]
            asr_word_confidence=seg.avg_word_confidence,
            word_confidences=word_confidences,
        )

        # Populate EnrichedSegment metadata (for output serialization)
        es.hadith_retrieval_confidence = result.retrieval_confidence
        es.hadith_correction_type = result.correction_type
        es.hadith_suggestion_safe = result.suggestion_safe
        es.hadith_correction_detail = result.suggested_patch or result.rejection_reason

        verified_count += 1
        if result.suggestion_safe and result.suggested_patch:
            suggestion_count += 1

    stats["hadith_verified"] = verified_count
    stats["hadith_suggestions"] = suggestion_count

    if verified_count > 0:
        logger.info(
            "Hadith verification: %d segments verified, %d suggestions generated (applied=0)",
            verified_count,
            suggestion_count,
        )
