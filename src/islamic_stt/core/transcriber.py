"""
transcriber.py
--------------
Wraps faster-whisper to transcribe an audio file and return a list of
structured segment objects with word-level timestamps.

Model   : Whisper large-v3
Device  : CUDA
Compute : float16   (falls back to int8 automatically on CPU for dev use)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from faster_whisper import WhisperModel
from tqdm import tqdm
from islamic_stt.core.arabic_utils import normalise_arabic, HALLUCINATION_PHRASES_NORMALISED

logger = logging.getLogger(__name__)


__all__ = ["Transcriber", "TranscriptSegment", "WordTimestamp"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WordTimestamp:
    """A single transcribed word with its timing and confidence."""
    word: str
    start: float   # seconds
    end: float     # seconds
    probability: float


@dataclass
class TranscriptSegment:
    """
    One contiguous speech segment as returned by faster-whisper.

    Fields
    ------
    id          : zero-based segment index
    start/end   : segment boundaries in seconds
    text        : raw transcribed text for this segment
    language    : ISO-639-1 code detected by Whisper for the *file* (e.g. 'en', 'ar', 'ur').
                  NOTE: this is file-level, not per-segment.  Per-segment detection
                  is handled by language_detector.py.
    language_probability : Whisper's confidence in the file-level language (0.0–1.0)
    avg_logprob : average log-probability of tokens in this segment (negative float;
                  lower = less confident transcription)
    words       : word-level timestamps (populated when word_timestamps=True)
    no_speech_prob : probability that the segment contains no speech
    avg_word_confidence : mean of word-level probabilities (0.0–1.0);
                          segments below ~0.35 indicate poor transcription quality
    """
    id: int
    start: float
    end: float
    text: str
    language: Optional[str]
    language_probability: float
    avg_logprob: float = 0.0
    words: List[WordTimestamp] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_word_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Loads faster-whisper large-v3 once and exposes a single `transcribe()`
    method that returns a list of TranscriptSegment objects.

    Parameters
    ----------
    model_size      : Whisper model identifier.  Defaults to 'large-v3'.
    device          : 'cuda' or 'cpu'.  Defaults to 'cuda'.
    compute_type    : 'float16' on GPU, 'int8' on CPU (cost-free quality
                      fallback so the same code runs on Colab and locally).
    download_root   : Optional directory to cache model weights.  If None,
                      faster-whisper uses its default cache (~/.cache/huggingface).
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: Optional[str] = None,
    ) -> None:
        # Automatically downgrade compute type on CPU to avoid an error.
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
        beam_size: int = 5,
        language: Optional[str] = "ur",
        vad_filter: bool = True,             # suppress silent regions
        vad_parameters: Optional[dict] = None,
        no_speech_threshold: float = 0.6,   # skip near-silent segments
        condition_on_previous_text: bool = False,  # False avoids error propagation (MED-4)
        initial_prompt: Optional[str] = None,
        log_prob_threshold: float = -1.0,              # drop low-confidence segments
        compression_ratio_threshold: float = 2.2,      # tighter repetition filter (was 2.4)
        hallucination_silence_threshold: Optional[float] = 2.0,  # skip hallucinations over silence
        repetition_penalty: float = 1.15,              # stronger anti-hallucination (was 1.08)
        no_repeat_ngram_size: int = 4,                 # wider dedup window (was 3)
    ) -> List[TranscriptSegment]:
        """
        Transcribe *audio_path* and return a list of TranscriptSegment objects.

        Parameters
        ----------
        audio_path : Path to an mp3 or wav file.
        beam_size  : Beam search width.  5 is the Whisper default.
        language   : Force a language (e.g. 'ur').  None = per-segment detection.
        vad_filter : Apply Silero VAD to skip silent chunks.
        no_speech_threshold : Segments with higher no-speech probability are dropped.
        condition_on_previous_text : Feed each segment's output as context for the next.
            Disabled by default to prevent error propagation in mixed-language content.
        initial_prompt : Optional text hint to guide the model (e.g. Islamic terminology).
        log_prob_threshold : Drop segments with average log probability below this value.
        compression_ratio_threshold : Drop segments with compression ratio above this value
            (indicates repetitive/hallucinated text).
        hallucination_silence_threshold : Skip segments generated over silence regions
            longer than this value in seconds.  Set to None to disable.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing: %s", audio_path)

        # Sensible VAD defaults optimised for lecture audio
        if vad_parameters is None:
            vad_parameters = {
                "min_silence_duration_ms": 300,   # tighter (was 500) — better segment boundaries
                "speech_pad_ms": 200,             # tighter (was 400) — reduces over-long segments
            }

        # faster-whisper returns a generator; we materialise it with a progress bar.
        segments_gen, info = self.model.transcribe(
            audio_path,
            task="transcribe",
            beam_size=beam_size,
            language=language,
            word_timestamps=True,         # always request word-level data
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
        )

        logger.info(
            "Audio duration: %.1fs | Detected language: %s (p=%.2f)",
            info.duration, info.language, info.language_probability,
        )

        # Collect segments with a tqdm progress bar keyed on audio duration.
        transcript: List[TranscriptSegment] = []
        with tqdm(
            total=round(info.duration),
            unit="s",
            desc="Transcribing",
            dynamic_ncols=True,
        ) as pbar:
            for raw_seg in segments_gen:
                # Build word list
                words: List[WordTimestamp] = []
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

                avg_wc = (
                    sum(w.probability for w in words) / len(words)
                    if words else 0.0
                )

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

                # Advance progress bar to segment end time
                pbar.update(max(0, round(raw_seg.end) - pbar.n))

        transcript = _deduplicate_segments(transcript)
        transcript = _drop_hallucination_phrases(transcript)
        from islamic_stt.core.segment_merger import merge_short_segments
        transcript = merge_short_segments(transcript)
        logger.info("Done — %d segment(s) after merge.", len(transcript))
        return transcript

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_timestamp(seconds: float, *, milliseconds: bool = False) -> str:
        """Convert a float second value to a timestamp string.

        Parameters
        ----------
        seconds      : Time in seconds (float).
        milliseconds : If True, return ``HH:MM:SS.mmm`` (SRT/VTT compatible).
                       If False (default), return ``HH:MM:SS``.
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if milliseconds:
            ms = int((seconds % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Hallucination deduplication (module-level, reusable)
# ---------------------------------------------------------------------------

def _deduplicate_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """
    Remove consecutive repeated segments caused by Whisper hallucination loops.

    Whisper sometimes gets stuck on a phrase and repeats it for tens of
    seconds (or even minutes) when audio is noisy, silent, or contains
    non-speech sounds.  This function collapses runs of identical text
    into a single segment, keeping the first occurrence.

    Two segments are considered duplicates when:
      - Their normalised text is identical (case/whitespace collapsed), AND
      - The later segment starts within DEDUP_WINDOW_SECONDS of the previous
        identical segment.

    A large window (120 s) is used so that the "إنما الأعمال بالنيات" loop
    seen in real lecture data — which ran for ~5 minutes — is caught, while
    legitimate repetitions in a poem or call-and-response separated by more
    than 2 minutes are preserved.

    Parameters
    ----------
    segments : Raw segment list from the transcription loop.

    Returns
    -------
    Deduplicated list with a log line for every collapsed run.
    """
    DEDUP_WINDOW_SECONDS = 120

    if not segments:
        return segments

    deduped: List[TranscriptSegment] = []
    # Track the last time we saw each normalised text string.
    last_seen: dict[str, float] = {}
    collapsed = 0

    for seg in segments:
        key = " ".join(seg.text.split()).lower()   # normalise whitespace + case
        last_time = last_seen.get(key)

        if last_time is not None and (seg.start - last_time) < DEDUP_WINDOW_SECONDS:
            # This is a duplicate within the window — skip it.
            collapsed += 1
            continue

        last_seen[key] = seg.start
        deduped.append(seg)

    if collapsed:
        logger.info(
            "Hallucination deduplication: removed %d repeated segment(s), %d remain.",
            collapsed, len(deduped),
        )

    return deduped



def _drop_hallucination_phrases(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    filtered: List[TranscriptSegment] = []
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
