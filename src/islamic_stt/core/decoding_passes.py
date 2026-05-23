"""
decoding_passes.py
------------------
Second-pass decoding helpers: contextual chunked transcription,
Arabic re-decode windows, and low-confidence segment retries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from islamic_stt.config import PipelineConfig
from islamic_stt.core.prompt_builder import (
    build_contextual_prompt,
    build_language_locked_prompt,
    detect_content_mode,
)
from islamic_stt.core.transcriber import Transcriber
from islamic_stt.core.types import TranscriptSegment

logger = logging.getLogger(__name__)

__all__ = [
    "transcribe_with_contextual_prompts",
    "apply_dual_pass_arabic",
    "retry_low_quality_segments",
    "segment_needs_retry",
]

_REPEAT_WORD_RE = re.compile(r"(?:\b(\S+)(?:\s+\1){2,})")


def segment_needs_retry(seg: TranscriptSegment, config: PipelineConfig) -> list[str]:
    """Returns a list of reasons if a segment needs retry, else empty list."""
    reasons = []
    if seg.avg_logprob < config.retry_logprob_threshold:
        reasons.append("low_logprob")
    if getattr(seg, "compression_ratio", 0.0) > getattr(config, "compression_ratio_threshold", 2.2):
        reasons.append("high_compression")
    if getattr(seg, "repeated_ngram_score", 0.0) > 0.35:
        reasons.append("repeated_ngram")
    if getattr(seg, "mixed_script_ratio", 0.0) > 0.15:
        reasons.append("mixed_script")
    return reasons


def _probe_duration(audio_path: str) -> float | None:
    try:
        import soundfile as sf

        return sf.info(audio_path).duration
    except Exception:
        return None


def transcribe_with_contextual_prompts(
    transcriber: Transcriber,
    audio_path: str,
    config: PipelineConfig,
) -> list[TranscriptSegment]:
    """
    Transcribe with mode-aware prompts. Long files use ~120s chunks;
    each chunk prompt is derived from the previous chunk's segments.
    """
    duration = _probe_duration(audio_path)
    lang = None if config.primary_language == "auto" else config.primary_language
    base_kwargs = dict(
        language=lang,
        beam_size=config.beam_size,
        best_of=config.best_of,
        patience=config.patience,
        no_speech_threshold=config.no_speech_threshold,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        compression_ratio_threshold=config.compression_ratio_threshold,
        log_prob_threshold=config.log_prob_threshold,
        hallucination_silence_threshold=config.hallucination_silence_threshold,
        temperature=config.temperature,
        condition_on_previous_text=config.condition_on_previous_text,
        vad_parameters={
            "min_silence_duration_ms": config.vad_min_silence_ms,
            "speech_pad_ms": config.vad_speech_pad_ms,
        },
    )

    use_chunks = (
        duration is not None
        and duration >= config.contextual_prompt_min_duration_s
        and config.contextual_prompt_chunk_s > 0
    )

    if not use_chunks:
        mode = detect_content_mode([])
        # Prompt Length Guidance
        max_tokens = 50
        if mode == "quran":
            max_tokens = 30
        elif mode == "bayan":
            max_tokens = 50
        prompt = build_contextual_prompt(
            mode,
            max_context_tokens=max_tokens,
            max_compression_ratio=config.compression_ratio_threshold,
        )
        return transcriber.transcribe(audio_path, initial_prompt=prompt, **base_kwargs)

    chunk_len = config.contextual_prompt_chunk_s
    overlap = float(config.chunk_overlap)
    all_segments: list[TranscriptSegment] = []
    mode = "bayan"
    t = 0.0
    seg_id = 0

    while t < duration:
        end_t = min(t + chunk_len, duration)
        mode = detect_content_mode(all_segments[-10:]) if all_segments else mode
        max_tokens = 50
        if mode == "quran":
            max_tokens = 30
        elif mode == "bayan":
            max_tokens = 50
        prompt = build_contextual_prompt(
            mode,
            all_segments[-5:],
            max_context_tokens=max_tokens,
            max_compression_ratio=config.compression_ratio_threshold,
        )
        chunk_segs = transcriber.transcribe_range(
            audio_path,
            t,
            end_t,
            initial_prompt=prompt,
            **base_kwargs,
        )
        for seg in chunk_segs:
            all_segments.append(
                replace(seg, id=seg_id),
            )
            seg_id += 1
        if end_t >= duration:
            break
        t = max(t + chunk_len - overlap, end_t)

    logger.info(
        "Contextual chunked transcription: %.0fs audio → %d segments (%d chunks).",
        duration or 0,
        len(all_segments),
        max(1, int((duration or chunk_len) / chunk_len)),
    )
    return all_segments


def _is_arabic_redraft_candidate(seg: TranscriptSegment) -> bool:
    from islamic_stt.core.arabic_utils import is_dominantly_arabic

    if seg.language == "ar":
        return True
    return is_dominantly_arabic(seg.text)


def _segment_confidence(seg: TranscriptSegment) -> float:
    speech_score = 1.0 - seg.no_speech_prob
    logprob_score = max(0.0, 1.0 + seg.avg_logprob)
    return round(0.25 * speech_score + 0.50 * logprob_score + 0.25, 4)


def apply_dual_pass_arabic(
    transcriber: Transcriber,
    audio_path: str,
    segments: list[TranscriptSegment],
    config: PipelineConfig,
) -> list[TranscriptSegment]:
    """Re-decode Arabic-dominant segments/windows with language=ar."""
    if not config.enable_dual_pass_arabic:
        return segments

    lang = "ar"
    prompt = build_language_locked_prompt(lang)
    redrafted = 0
    out = list(segments)

    for seg in out:
        if not _is_arabic_redraft_candidate(seg):
            continue
        duration = seg.end - seg.start
        if duration > config.dual_pass_max_window_s or duration < 0.25:
            continue

        pad = 0.15
        try:
            new_segs = transcriber.transcribe_range(
                audio_path,
                max(0.0, seg.start - pad),
                seg.end + pad,
                language=lang,
                initial_prompt=prompt,
                beam_size=config.beam_size,
                best_of=config.best_of,
                patience=config.patience,
                no_speech_threshold=config.no_speech_threshold,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                compression_ratio_threshold=2.2,
                log_prob_threshold=config.log_prob_threshold,
                hallucination_silence_threshold=config.hallucination_silence_threshold,
                temperature=(0.0, 0.1),
                condition_on_previous_text=False,
                vad_parameters={
                    "min_silence_duration_ms": config.vad_min_silence_ms,
                    "speech_pad_ms": config.vad_speech_pad_ms,
                },
            )
        except Exception as exc:
            logger.debug("Arabic redraft failed for seg %s: %s", seg.id, exc)
            continue

        if not new_segs:
            continue

        new_text = " ".join(s.text.strip() for s in new_segs if s.text.strip())
        if not new_text or len(new_text) < 3:
            continue

        idx = next(i for i, s in enumerate(out) if s.id == seg.id)
        out[idx] = replace(
            out[idx],
            text=new_text,
            language=lang,
            avg_logprob=sum(s.avg_logprob for s in new_segs) / len(new_segs),
            avg_word_confidence=(sum(s.avg_word_confidence for s in new_segs) / len(new_segs)),
        )
        redrafted += 1

    if redrafted:
        logger.info("Arabic dual-pass: re-decoded %d segment(s).", redrafted)
    return out, redrafted


def retry_low_quality_segments(
    transcriber: Transcriber,
    audio_path: str,
    segments: list[TranscriptSegment],
    config: PipelineConfig,
) -> tuple[list[TranscriptSegment], int]:
    """Re-decode weak segments with safer, higher-beam settings."""
    if not config.enable_retry_decoding:
        return segments, 0

    lang = None if config.primary_language == "auto" else config.primary_language
    out = list(segments)
    retried = 0

    for seg in out:
        reasons = segment_needs_retry(seg, config)
        if not reasons:
            continue

        pad = 0.1
        try:
            candidates = transcriber.transcribe_range(
                audio_path,
                max(0.0, seg.start - pad),
                seg.end + pad,
                language=lang,
                beam_size=config.beam_size + config.retry_beam_boost,
                best_of=max(config.best_of, 5),
                patience=config.patience,
                no_speech_threshold=config.no_speech_threshold,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                hallucination_silence_threshold=config.hallucination_silence_threshold,
                temperature=(0.0,),
                condition_on_previous_text=False,
                suppress_blank=True,
                without_timestamps=False,
                vad_parameters={
                    "min_silence_duration_ms": config.vad_min_silence_ms,
                    "speech_pad_ms": config.vad_speech_pad_ms,
                },
            )
        except Exception as exc:
            logger.debug("Retry decode failed for seg %s: %s", seg.id, exc)
            continue

        if not candidates:
            continue

        new_text = " ".join(c.text.strip() for c in candidates if c.text.strip())
        if not new_text:
            continue

        new_seg = replace(
            seg,
            text=new_text,
            avg_logprob=sum(c.avg_logprob for c in candidates) / len(candidates),
            avg_word_confidence=(sum(c.avg_word_confidence for c in candidates) / len(candidates)),
            prompt_safe=False,
            retry_triggered=True,
            retry_reason=reasons,
            original_text=seg.text,
        )
        old_conf = _segment_confidence(seg)
        new_conf = _segment_confidence(new_seg)
        if new_conf <= old_conf:
            continue

        new_seg.confidence_delta = new_conf - old_conf

        idx = next(i for i, s in enumerate(out) if s.id == seg.id)
        out[idx] = new_seg
        retried += 1

    if retried:
        logger.info("Retry decoding: improved %d segment(s).", retried)
    return out, retried
