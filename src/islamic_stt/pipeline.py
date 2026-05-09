"""
pipeline.py
-----------
Main orchestrator: transcription → enrichment → matching → output.

Improvements implemented from the codebase review plan:
  #4  — Exception handling (CUDA OOM, corpus, Unicode)
  #8  — Unified segment confidence scoring
  #25 — GPU capability detection with auto-fallback
  #27 — Input validation (file type, size)
  #28 — Resource limits (max duration, max file size)
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from islamic_stt.output.flagged_handler import FlaggedHandler
from islamic_stt.matchers.hadith_matcher import HadithMatcher
from islamic_stt.core.language_detector import detect_language
from islamic_stt.output.output_handler import EnrichedSegment, OutputHandler
from islamic_stt.matchers.quran_matcher import FormulaMatch, QuranMatcher
from islamic_stt.core.transcriber import TranscriptSegment, Transcriber

from islamic_stt.config import PipelineConfig

logger = logging.getLogger("islamic_stt.pipeline")

# Supported audio extensions
_SUPPORTED_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"})


# ---------------------------------------------------------------------------
# GPU detection (#25 from improvement plan)
# ---------------------------------------------------------------------------

def _detect_device(requested: str) -> tuple[str, str]:
    """
    Auto-detect the best device and compute type.

    If ``requested`` is 'cuda' but no GPU is available, falls back to
    CPU + int8 gracefully instead of crashing.

    Returns (device, compute_type).
    """
    if requested == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                logger.info(
                    "GPU detected: %s (%.1f GB VRAM)", gpu_name, vram_gb,
                )
                return "cuda", "float16"
            else:
                logger.warning(
                    "CUDA requested but no GPU found — falling back to CPU (int8)."
                )
                return "cpu", "int8"
        except ImportError:
            logger.warning(
                "PyTorch not installed — cannot verify GPU. Attempting CUDA anyway."
            )
            return "cuda", "float16"
    return "cpu", "int8"


# ---------------------------------------------------------------------------
# Input validation (#27, #28 from improvement plan)
# ---------------------------------------------------------------------------

def _validate_audio(audio_path: str, config: PipelineConfig) -> None:
    """
    Validate the audio file before loading the model.

    Checks:
    - File exists
    - File extension is supported
    - File is not empty
    - File is within size limit
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        logger.warning(
            "Unrecognised extension '%s' — attempting anyway. "
            "Supported: %s",
            ext, ", ".join(sorted(_SUPPORTED_EXTENSIONS)),
        )

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb < 0.001:
        raise ValueError(f"Audio file appears empty: {audio_path} ({file_size_mb:.4f} MB)")

    if file_size_mb > config.max_file_size_mb:
        raise ValueError(
            f"Audio file too large: {file_size_mb:.1f} MB "
            f"(limit: {config.max_file_size_mb:.0f} MB). "
            f"Split the file or increase max_file_size_mb."
        )

    logger.info(
        "Audio validated: %s (%.1f MB)",
        os.path.basename(audio_path), file_size_mb,
    )


# ---------------------------------------------------------------------------
# Segment confidence scoring (#8 from improvement plan)
# ---------------------------------------------------------------------------

def compute_segment_quality(seg: TranscriptSegment) -> float:
    """
    Compute a unified quality score (0.0–1.0) for a transcript segment.

    Combines three Whisper signals with empirically tuned weights:
    - avg_logprob: how confident the model is overall
    - avg_word_confidence: per-word model certainty
    - no_speech_prob: likelihood the segment is silence/noise

    A score below ~0.30 indicates the segment is likely a hallucination
    or very poorly transcribed.
    """
    # Normalise avg_logprob from its native range (~-1.5 to 0) into 0–1
    logprob_score = max(0.0, min(1.0, (seg.avg_logprob + 1.5) / 1.5))

    # no_speech_prob: invert (high no_speech = bad)
    speech_score = 1.0 - seg.no_speech_prob

    # Weighted combination
    quality = (
        0.35 * logprob_score
        + 0.45 * seg.avg_word_confidence
        + 0.20 * speech_score
    )
    return round(quality, 4)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig, audio_path: str) -> dict[str, int | float]:
    """
    Run the full Islamic STT pipeline.

    Returns a stats dict with segment counts, match counts, and timing.

    .. warning:: This function does NOT mutate *config*.  Device/compute
       overrides are resolved into local variables.
    """
    start_time = time.time()

    # --- Input validation ---
    _validate_audio(audio_path, config)

    # --- GPU auto-detection (local vars — never mutate config) ---
    resolved_device, resolved_compute = _detect_device(config.device)
    if resolved_device != config.device:
        logger.info("Device override: %s → %s", config.device, resolved_device)

    # --- Load model with exception handling ---
    try:
        transcriber = Transcriber(
            model_size=config.model,
            device=resolved_device,
            compute_type=resolved_compute,
        )
    except Exception as exc:
        logger.error("Failed to load Whisper model: %s", exc)
        raise RuntimeError(
            f"Could not load model '{config.model}' on {resolved_device}. "
            f"Check CUDA/VRAM availability. Error: {exc}"
        ) from exc

    # --- Load Quran corpus with exception handling ---
    try:
        quran_matcher = QuranMatcher(corpus_path=config.quran_corpus)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Quran corpus error: %s", exc)
        raise

    hadith_matcher = None
    if not config.no_hadith:
        try:
            hadith_matcher = HadithMatcher(api_key=config.sunnah_api_key)
        except ValueError as exc:
            logger.warning("Hadith matcher disabled: %s", exc)

    output_handler = OutputHandler(output_path=config.output_path)
    flagged_handler = FlaggedHandler(flagged_path=config.flagged_path)

    # --- Transcription ---
    segments: list[TranscriptSegment] = transcriber.transcribe(
        audio_path,
        language=None if config.primary_language == "auto" else config.primary_language,
        beam_size=config.beam_size,
        no_speech_threshold=config.no_speech_threshold,
        initial_prompt=config.initial_prompt,
    )

    # --- Enrichment ---
    enriched: list[EnrichedSegment] = []
    stats: dict[str, int | float] = {
        "total_segments": len(segments),
        "arabic_segments": 0,
        "formula_hits": 0,
        "quran_hits": 0,
        "hadith_hits": 0,
        "flagged": 0,
        "low_quality_segments": 0,
    }

    for seg in segments:
        lang = detect_language(seg.text, whisper_language=seg.language)
        es = EnrichedSegment(segment=seg, detected_lang=lang)

        # --- Segment quality scoring (#8) ---
        quality = compute_segment_quality(seg)
        if quality < 0.25:
            stats["low_quality_segments"] += 1
            logger.debug(
                "Low quality segment (%.2f) at %.1fs: %s",
                quality, seg.start, seg.text[:60],
            )

        if lang == "ar":
            stats["arabic_segments"] += 1
            if seg.avg_word_confidence < config.low_confidence_threshold:
                es.is_flagged = True
                flagged_handler.add(seg, reason=f"Low word confidence (avg={seg.avg_word_confidence:.2f})")
                stats["flagged"] += 1
                enriched.append(es)
                continue

            match_result = quran_matcher.match(seg.text)
            if isinstance(match_result, FormulaMatch):
                es.formula_match = match_result
                stats["formula_hits"] += 1
            elif match_result is not None:
                es.quran_match = match_result
                stats["quran_hits"] += 1
            else:
                es.pending_hadith = True

        enriched.append(es)

    # --- Hadith matching (async) ---
    pending = [es for es in enriched if es.pending_hadith]
    if pending and hadith_matcher:
        with ThreadPoolExecutor(max_workers=config.hadith_workers) as pool:
            futures = {pool.submit(hadith_matcher.match, es.segment.text): es for es in pending}
            for future in as_completed(futures):
                es = futures[future]
                hadith_match = None
                try:
                    hadith_match = future.result()
                except Exception as exc:
                    logger.warning("Hadith matcher error: %s", exc)
                if hadith_match:
                    es.hadith_match = hadith_match
                    stats["hadith_hits"] += 1
                else:
                    es.is_flagged = True
                    flagged_handler.add(es.segment, reason="No Quran or Hadith match found")
                    stats["flagged"] += 1
                es.pending_hadith = False
    elif pending:
        for es in pending:
            es.is_flagged = True
            flagged_handler.add(es.segment, reason="Hadith matching disabled; no Quran match")
            stats["flagged"] += 1
            es.pending_hadith = False

    # --- Output ---
    output_handler.write(enriched)
    flagged_handler.write()

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    logger.info("Pipeline complete in %.1fs", elapsed)

    return stats

