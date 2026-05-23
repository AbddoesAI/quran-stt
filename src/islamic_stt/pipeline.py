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
import re
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from islamic_stt.core.arabic_utils import normalise_arabic, contains_arabic_script
from islamic_stt.core.language_detector import detect_language
from islamic_stt.core.post_processor import apply_post_processing
from islamic_stt.core.segment_merger import merge_short_segments
from islamic_stt.core.types import TranscriptSegment, WordTimestamp
from islamic_stt.core.transcriber import Transcriber
from islamic_stt.matchers.quran_matcher import FormulaMatch, QuranMatcher
from islamic_stt.output.flagged_handler import FlaggedHandler
from islamic_stt.output.output_handler import EnrichedSegment, OutputHandler
from islamic_stt.config import PipelineConfig

logger = logging.getLogger("islamic_stt.pipeline")

_SUPPORTED_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm",
})

# Urdu-exclusive codepoints (Urdu-specific letters not used in Arabic)
# Used to filter Arabic span extraction
_URDU_EXCLUSIVE = frozenset("ٹپچڈڑژکگںھہۃیے")


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
                vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
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
            os.path.basename(audio_path), file_size_mb, duration_s,
        )
    else:
        logger.info("Audio validated: %s (%.1f MB, duration unknown)", os.path.basename(audio_path), file_size_mb)


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
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def compute_segment_quality(seg: TranscriptSegment) -> float:
    logprob_score = max(0.0, min(1.0, (seg.avg_logprob + 1.5) / 1.5))
    speech_score = 1.0 - seg.no_speech_prob
    return round(
        0.35 * logprob_score + 0.45 * seg.avg_word_confidence + 0.20 * speech_score,
        4,
    )


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

# Regex: contiguous Arabic-script runs with internal spaces
_ARABIC_SPAN_RE = re.compile(
    r'([\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]'
    r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\u0640]{8,})'
)


def _extract_arabic_spans(text: str) -> list[str]:
    """
    Extract contiguous Arabic-script spans from mixed Urdu/Arabic text.

    Review 3 fix: tokenize first, then build spans from consecutive
    non-Urdu tokens. This handles 'Urdu + Quran' as one continuous
    Arabic-script run without discarding the entire run when a single
    Urdu character appears.
    """
    # Split into whitespace-delimited tokens
    tokens = text.split()
    spans: list[str] = []
    current_arabic: list[str] = []

    for token in tokens:
        # Check if this token is Arabic-script
        if not any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF' for c in token):
            # Non-Arabic token: flush any accumulated Arabic tokens
            if len(current_arabic) >= 3:
                spans.append(" ".join(current_arabic))
            current_arabic = []
            continue

        # Check if token contains Urdu-exclusive characters
        has_urdu = any(c in _URDU_EXCLUSIVE for c in token)

        if has_urdu:
            # Urdu token: flush accumulated Arabic tokens
            if len(current_arabic) >= 3:
                spans.append(" ".join(current_arabic))
            current_arabic = []
        else:
            # Arabic token: accumulate
            current_arabic.append(token)

    # Flush remaining
    if len(current_arabic) >= 3:
        spans.append(" ".join(current_arabic))

    # Filter: require at least one classical Arabic signal
    filtered: list[str] = []
    for span in spans:
        has_diacritics = bool(re.search(r'[\u064B-\u065F\u0670]', span))
        has_hamza = bool(re.search(r'[ؤئأإ]', span))
        has_classical = bool(re.search(r'[ةى]', span))
        word_count = len(span.split())
        if has_diacritics or has_hamza or has_classical or word_count >= 5:
            filtered.append(span)

    return filtered


# ---------------------------------------------------------------------------
# Hadith matcher loading
# ---------------------------------------------------------------------------

def _load_hadith_matcher(config: PipelineConfig):
    """Load Hadith matcher: prefer local DB, fall back to API."""
    if os.path.isfile(config.hadith_db_path):
        try:
            from islamic_stt.matchers.hadith_db import LocalHadithMatcher
            matcher = LocalHadithMatcher(db_path=config.hadith_db_path)
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

    # --- GPU auto-detection ---
    resolved_device, resolved_compute = _detect_device(config)
    if resolved_device != config.device or resolved_compute != config.compute_type:
        logger.info(
            "Device/compute: %s/%s → %s/%s",
            config.device, config.compute_type, resolved_device, resolved_compute,
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
    # STEP 1: Transcribe
    # ================================================================
    with _timed("transcription", stats):
        segments: list[TranscriptSegment] = transcriber.transcribe(
            audio_path,
            language=None if config.primary_language == "auto" else config.primary_language,
            beam_size=config.beam_size,
            best_of=config.best_of,
            patience=config.patience,
            no_speech_threshold=config.no_speech_threshold,
            initial_prompt=config.initial_prompt,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            compression_ratio_threshold=config.compression_ratio_threshold,
            log_prob_threshold=config.log_prob_threshold,
            hallucination_silence_threshold=config.hallucination_silence_threshold,
            temperature=config.temperature,
            vad_parameters={
                "min_silence_duration_ms": config.vad_min_silence_ms,
                "speech_pad_ms": config.vad_speech_pad_ms,
            },
        )
    stats["total_segments"] = len(segments)

    # ================================================================
    # STEP 2: Classify language per segment (before merge!)
    # ================================================================
    with _timed("classify", stats):
        seg_langs: list[str] = []
        for seg in segments:
            lang = _classify_segment_language(seg)
            seg_langs.append(lang)
            # Stamp detected language onto segment for merger boundary checks
            seg.language = lang  # type: ignore[misc]

    # ================================================================
    # STEP 3: Merge short segments (using detected language)
    # ================================================================
    with _timed("merge", stats):
        merged_segments = merge_short_segments(segments)
        # Re-classify merged segments (text may have changed)
        merged_langs: list[str] = []
        for seg in merged_segments:
            lang = _classify_segment_language(seg)
            merged_langs.append(lang)

    logger.info("Segments: %d raw → %d after merge.", len(segments), len(merged_segments))

    # ================================================================
    # STEP 4: Post-process with raw provenance
    # ================================================================
    with _timed("post_process", stats):
        raw_texts: list[str] = []  # keep original for auditability
        for seg in merged_segments:
            raw_texts.append(seg.text)
            corrected = apply_post_processing(seg.text)
            if corrected != seg.text:
                stats["post_processed"] = stats.get("post_processed", 0) + 1
                seg.text = corrected  # type: ignore[misc]

    # ================================================================
    # STEP 5: Verify (Quran + Hadith) — runs on final merged text
    # ================================================================
    with _timed("enrichment", stats):
        enriched: list[EnrichedSegment] = []
        for i, (seg, lang, raw_text) in enumerate(
            zip(merged_segments, merged_langs, raw_texts)
        ):
            was_corrected = (seg.text != raw_text)
            es = EnrichedSegment(
                segment=seg,
                detected_lang=lang,
                raw_text=raw_text,
                was_corrected=was_corrected,
            )

            quality = compute_segment_quality(seg)
            if quality < config.low_quality_threshold:
                stats["low_quality_segments"] += 1

            if lang == "ar":
                stats["arabic_segments"] += 1
                if seg.avg_word_confidence < config.low_confidence_threshold:
                    es.is_flagged = True
                    flagged_handler.add(
                        seg, reason=f"Low confidence (avg={seg.avg_word_confidence:.2f})"
                    )
                    stats["flagged"] += 1
                    enriched.append(es)
                    continue

                norm_text = normalise_arabic(seg.text)
                match_result = quran_matcher.match(seg.text, normalised=norm_text)

                if isinstance(match_result, FormulaMatch):
                    es.formula_match = match_result
                    stats["formula_hits"] += 1
                elif match_result is not None:
                    # P0 review 3: If match depended on post-processing correction,
                    # ALWAYS downgrade — even exact matches. A corrected exact match
                    # must never render as '✓exact (100%)'.
                    if was_corrected:
                        match_result = match_result.__class__(
                            surah_id=match_result.surah_id,
                            surah_name=match_result.surah_name,
                            ayah_id=match_result.ayah_id,
                            original_text=match_result.original_text,
                            matched_text=raw_text,  # original ASR output
                            confidence=match_result.confidence * 0.90,
                            is_exact=False,  # NEVER exact if text was corrected
                            is_ambiguous=getattr(match_result, 'is_ambiguous', False),
                            ambiguous_count=getattr(match_result, 'ambiguous_count', 1),
                        )
                    es.quran_match = match_result
                    stats["quran_hits"] += 1
                else:
                    es.pending_hadith = True

            elif lang == "ur" and contains_arabic_script(seg.text):
                # Extract Arabic spans from Urdu segments
                arabic_spans = _extract_arabic_spans(seg.text)
                for span in arabic_spans:
                    norm_span = normalise_arabic(span)
                    span_match = quran_matcher.match(span, normalised=norm_span)
                    if isinstance(span_match, FormulaMatch):
                        es.formula_match = span_match
                        stats["formula_hits"] += 1
                        break
                    elif span_match is not None:
                        # P0 review 4: Apply same correction penalty to span matches
                        if was_corrected:
                            span_match = span_match.__class__(
                                surah_id=span_match.surah_id,
                                surah_name=span_match.surah_name,
                                ayah_id=span_match.ayah_id,
                                original_text=span_match.original_text,
                                matched_text=span,  # the extracted span, not full segment
                                confidence=span_match.confidence * 0.90,
                                is_exact=False,
                                is_ambiguous=getattr(span_match, 'is_ambiguous', False),
                                ambiguous_count=getattr(span_match, 'ambiguous_count', 1),
                                alternate_refs=getattr(span_match, 'alternate_refs', None),
                            )
                        es.quran_match = span_match
                        stats["quran_hits"] += 1
                        break
                if arabic_spans and not es.quran_match and not es.formula_match:
                    es.pending_hadith = True
                    stats["arabic_spans_in_urdu"] = stats.get("arabic_spans_in_urdu", 0) + 1

            enriched.append(es)

    # ================================================================
    # STEP 6: Hadith matching
    # ================================================================
    pending = [es for es in enriched if es.pending_hadith]
    if pending and hadith_matcher:
        with _timed("hadith", stats):
            _run_hadith_matching(
                pending, hadith_matcher, hadith_is_local,
                flagged_handler, stats, config,
            )
    elif pending:
        for es in pending:
            es.is_flagged = True
            flagged_handler.add(es.segment, reason="Hadith matching disabled; no Quran match")
            stats["flagged"] += 1
            es.pending_hadith = False

    # ================================================================
    # STEP 7: Output
    # ================================================================
    with _timed("output", stats):
        output_handler.write(enriched)
        flagged_handler.write()

    if hadith_matcher and hasattr(hadith_matcher, "close"):
        hadith_matcher.close()

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    timing = ", ".join(f"{k}={v}ms" for k, v in stats.items() if k.endswith("_ms"))
    logger.info("Pipeline complete in %.1fs | %s", elapsed, timing)
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
        for es, hm in zip(pending, results):
            _apply_hadith_result(es, hm, flagged_handler, stats)
    except Exception as exc:
        logger.warning("Async Hadith failed: %s — ThreadPool fallback.", exc)
        with ThreadPoolExecutor(max_workers=config.hadith_workers) as pool:
            futures = {pool.submit(hadith_matcher.match, es.segment.text): es for es in pending}  # type: ignore
            for future in as_completed(futures):
                es = futures[future]
                hm = None
                try:
                    hm = future.result()
                except Exception:
                    pass
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
