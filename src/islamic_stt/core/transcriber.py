"""
transcriber.py
--------------
Wraps faster-whisper to transcribe an audio file and return a list of
structured segment objects with word-level timestamps.

Model   : Whisper large-v3
Device  : CUDA
Compute : float16   (falls back to int8 automatically on CPU for dev use)

Changes (accuracy overhaul)
---------------------------
- CRIT-3 : beam_size=8, best_of=5, patience=1.5 for better multilingual decoding
- CRIT-5 : Dedup window reduced 120s → 8s; Islamic formulas whitelisted from dedup
- CRIT-6 : Pattern-based hallucination filtering (repeated words, URLs, music markers)
- Tuned  : temperature fallback, stronger compression/logprob thresholds
- Tuned  : VAD parameters for better sentence-level segmentation
"""

from __future__ import annotations

import logging
import os
import re

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used in minimal test environments

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def update(self, n: int) -> None:
            self.n += n


from islamic_stt.core.arabic_utils import (
    HALLUCINATION_PATTERNS,
    HALLUCINATION_PHRASES_NORMALISED,
    normalise_arabic,
)

# P0 fix: Import shared types from types.py to break circular import
from islamic_stt.core.types import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)

__all__ = ["Transcriber", "TranscriptSegment", "WordTimestamp"]

# Precompiled whitespace collapse regex
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------


class Transcriber:
    """
    Loads faster-whisper large-v3 once and exposes a single `transcribe()`
    method that returns a list of TranscriptSegment objects.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required for transcription. Install the project "
                "dependencies before running the STT pipeline."
            ) from exc

        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            logger.info("CPU detected — switching compute_type to int8.")

        logger.info("Loading model '%s' on %s (%s) …", model_size, device, compute_type)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        self.model_size = model_size
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        *,
        beam_size: int = 8,
        best_of: int = 5,
        patience: float = 1.5,
        language: str | None = None,
        vad_filter: bool = True,
        vad_parameters: dict | None = None,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        initial_prompt: str | None = None,
        log_prob_threshold: float = -0.8,
        compression_ratio_threshold: float = 2.0,
        hallucination_silence_threshold: float | None = 1.0,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        temperature: tuple[float, ...] | float = (0.0, 0.2, 0.4, 0.6),
    ) -> list[TranscriptSegment]:
        """
        Transcribe *audio_path* and return a list of TranscriptSegment objects.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing: %s", audio_path)

        if vad_parameters is None:
            vad_parameters = {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            }

        segments_gen, info = self.model.transcribe(
            audio_path,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            language=language,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=list(temperature) if isinstance(temperature, tuple) else temperature,
        )

        logger.info(
            "Audio duration: %.1fs | Detected language: %s (p=%.2f)",
            info.duration,
            info.language,
            info.language_probability,
        )

        transcript: list[TranscriptSegment] = []
        with tqdm(
            total=round(info.duration),
            unit="s",
            desc="Transcribing",
            dynamic_ncols=True,
        ) as pbar:
            for raw_seg in segments_gen:
                words: list[WordTimestamp] = []
                if raw_seg.words:
                    for w in raw_seg.words:
                        words.append(
                            WordTimestamp(
                                word=w.word,
                                start=w.start,
                                end=w.end,
                                probability=w.probability,
                            )
                        )

                avg_wc = sum(w.probability for w in words) / len(words) if words else 0.0

                seg = TranscriptSegment(
                    id=raw_seg.id,
                    start=raw_seg.start,
                    end=raw_seg.end,
                    text=raw_seg.text.strip(),
                    language=info.language,
                    language_probability=info.language_probability,
                    avg_logprob=raw_seg.avg_logprob,
                    words=words,
                    no_speech_prob=raw_seg.no_speech_prob,
                    avg_word_confidence=avg_wc,
                )
                transcript.append(seg)
                pbar.update(max(0, round(raw_seg.end) - pbar.n))

        transcript = _deduplicate_segments(transcript)
        transcript = _drop_hallucination_phrases(transcript)
        transcript = _drop_hallucination_patterns(transcript)
        # NOTE: merge_short_segments is called by pipeline.py AFTER
        # per-segment language detection, so merging respects actual
        # detected language instead of file-level Whisper language.
        logger.info("Done — %d segment(s) after cleanup.", len(transcript))
        return transcript

    def transcribe_range(
        self,
        audio_path: str,
        start_s: float,
        end_s: float,
        *,
        beam_size: int = 8,
        best_of: int = 5,
        patience: float = 1.5,
        language: str | None = None,
        vad_filter: bool = True,
        vad_parameters: dict | None = None,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        initial_prompt: str | None = None,
        log_prob_threshold: float = -0.8,
        compression_ratio_threshold: float = 2.0,
        hallucination_silence_threshold: float | None = 1.0,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        temperature: tuple[float, ...] | float = (0.0, 0.2),
    ) -> list[TranscriptSegment]:
        """
        Transcribe a time slice of *audio_path* (seconds). Segment timestamps
        are offset to absolute file time.
        """
        start_s = max(0.0, start_s)
        end_s = max(start_s + 0.05, end_s)
        clip = f"{start_s:.3f},{end_s:.3f}"

        if vad_parameters is None:
            vad_parameters = {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            }

        segments_gen, info = self.model.transcribe(
            audio_path,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            language=language,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=list(temperature) if isinstance(temperature, tuple) else temperature,
            clip_timestamps=clip,
        )

        transcript: list[TranscriptSegment] = []
        for raw_seg in segments_gen:
            words: list[WordTimestamp] = []
            if raw_seg.words:
                for w in raw_seg.words:
                    words.append(
                        WordTimestamp(
                            word=w.word,
                            start=start_s + w.start,
                            end=start_s + w.end,
                            probability=w.probability,
                        )
                    )
            avg_wc = sum(w.probability for w in words) / len(words) if words else 0.0
            transcript.append(
                TranscriptSegment(
                    id=raw_seg.id,
                    start=start_s + raw_seg.start,
                    end=start_s + raw_seg.end,
                    text=raw_seg.text.strip(),
                    language=info.language,
                    language_probability=info.language_probability,
                    avg_logprob=raw_seg.avg_logprob,
                    words=words,
                    no_speech_prob=raw_seg.no_speech_prob,
                    avg_word_confidence=avg_wc,
                )
            )

        transcript = _deduplicate_segments(transcript)
        transcript = _drop_hallucination_phrases(transcript)
        transcript = _drop_hallucination_patterns(transcript)
        return transcript

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_timestamp(seconds: float, *, milliseconds: bool = False) -> str:
        """Convert a float second value to a timestamp string."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if milliseconds:
            ms = int((seconds % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Islamic formula whitelist — these are legitimately repeated in lectures
# ---------------------------------------------------------------------------
_ISLAMIC_FORMULA_DEDUP_WHITELIST: frozenset[str] = frozenset(
    {
        normalise_arabic(p)
        for p in [
            "صلى الله عليه وسلم",
            "سبحان الله",
            "الحمد لله",
            "الله اكبر",
            "لا اله الا الله",
            "استغفر الله",
            "رضي الله عنه",
            "رضي الله عنها",
            "رحمه الله",
            "سبحانه وتعالى",
            "عز وجل",
            "ان شاء الله",
            "ما شاء الله",
            "بسم الله الرحمن الرحيم",
            "جزاك الله خيرا",
            "بارك الله فيك",
            "لا حول ولا قوة الا بالله",
            "حسبنا الله ونعم الوكيل",
            "انا لله وانا اليه راجعون",
        ]
    }
)


# ---------------------------------------------------------------------------
# Hallucination deduplication (CRIT-5 fix: 120s → 8s + whitelist)
# ---------------------------------------------------------------------------


def _make_dedup_key(text: str) -> str:
    """Normalise whitespace and case for dedup comparison."""
    return _WS_RE.sub(" ", text).strip().casefold()


def _deduplicate_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """
    Remove consecutive repeated segments caused by Whisper hallucination loops.

    CRIT-5 fix: Reduced window from 120s to 8s to prevent deleting
    legitimate repeated Islamic phrases (dhikr, salawat, takbir).
    Islamic formulas are whitelisted from deduplication entirely.
    """
    DEDUP_WINDOW_SECONDS = 8  # was 120 — far too aggressive

    if not segments:
        return segments

    deduped: list[TranscriptSegment] = []
    last_seen: dict[str, float] = {}
    collapsed = 0

    for seg in segments:
        key = _make_dedup_key(seg.text)
        norm = normalise_arabic(seg.text)

        # Never dedup Islamic formulas — they are legitimately repeated
        if norm in _ISLAMIC_FORMULA_DEDUP_WHITELIST:
            deduped.append(seg)
            continue

        last_time = last_seen.get(key)

        if last_time is not None and (seg.start - last_time) < DEDUP_WINDOW_SECONDS:
            # Additional check: only dedup short segments (< 3 seconds)
            # Longer segments with same text are likely real content
            seg_duration = seg.end - seg.start
            if seg_duration < 3.0:
                collapsed += 1
                continue

        last_seen[key] = seg.start
        deduped.append(seg)

    if collapsed:
        logger.info(
            "Hallucination deduplication: removed %d repeated segment(s), %d remain.",
            collapsed,
            len(deduped),
        )

    return deduped


def _drop_hallucination_phrases(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Drop segments that exactly match known hallucination phrases."""
    filtered: list[TranscriptSegment] = []
    dropped = 0
    for seg in segments:
        norm = normalise_arabic(seg.text)
        if norm in HALLUCINATION_PHRASES_NORMALISED:
            dropped += 1
            continue
        filtered.append(seg)
    if dropped:
        logger.info("Filtered %d known hallucination segment(s).", dropped)
    return filtered


def _drop_hallucination_patterns(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """
    Drop segments matching structural hallucination patterns.
    (CRIT-6: pattern-based filtering for repeated words, music markers, URLs)
    """
    filtered: list[TranscriptSegment] = []
    dropped = 0
    for seg in segments:
        text = seg.text.strip()
        # Skip empty/whitespace-only
        if not text:
            dropped += 1
            continue
        # Check against structural patterns
        is_hallucination = False
        for pattern in HALLUCINATION_PATTERNS:
            if pattern.search(text):
                is_hallucination = True
                break
        if is_hallucination:
            logger.debug("Dropped hallucination pattern: %s", text[:60])
            dropped += 1
            continue
        filtered.append(seg)
    if dropped:
        logger.info("Filtered %d pattern-matched hallucination segment(s).", dropped)
    return filtered
