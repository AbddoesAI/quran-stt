from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Literal

ProfileName = Literal["default", "colab-fast", "colab-accurate", "urdu-bayan", "quran-recitation"]


# ---------------------------------------------------------------------------
# Domain-dense initial prompt — fills Whisper's 224-token context window
# ---------------------------------------------------------------------------
# This prompt serves three critical purposes:
#   1. Urdu instruction prevents Whisper from translating to English
#   2. Diacritized Arabic primes the decoder for tashkeel-aware output
#   3. Named entities (narrators, collections, surahs) inject domain vocabulary
#      into the decoder's token distribution, dramatically improving recognition
#      of Islamic terminology.

DEFAULT_INITIAL_PROMPT = (
    # Urdu instruction: transcribe as spoken, do not translate
    "یہ ایک اسلامی بیان ہے جس میں اردو، عربی اور انگریزی تینوں زبانیں بولی جاتی ہیں۔ "
    "ہر زبان کو اسی زبان میں لکھیں، ترجمہ نہ کریں۔ "
    # Quranic Arabic WITH diacritics (primes decoder for tashkeel output)
    "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ۔ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ۔ "
    "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ۔ سُبْحَانَ اللَّهِ۔ لَا إِلَهَ إِلَّا اللَّهُ۔ "
    # Narrator names (critical for Hadith detection downstream)
    "ابو ہریرہ، عبداللہ بن عمر، حضرت عائشہ، حضرت انس بن مالک، ابن عباس۔ "
    # Collection names in Urdu script (as speakers actually say them)
    "صحیح بخاری، صحیح مسلم، سنن ترمذی، سنن ابو داؤد، سنن نسائی، سنن ابن ماجہ۔ "
    # Common Islamic terms the speaker uses in Urdu context
    "قرآن، حدیث، سنت، صحابہ، تابعین، تفسیر، فقہ، اجتہاد، شریعت، عقیدہ۔ "
    # Surah names (both Arabic and as spoken in Urdu lectures)
    "سورة البقرة، سورة آل عمران، سورة النساء، سورة المائدة، يس، الرحمن، الملك۔ "
    # English terms commonly heard in Islamic lectures
    "Islamic, Quran, Hadith, Sunnah, Prophet Muhammad, peace be upon him. "
    "Sahih Bukhari, Sahih Muslim, Surah, Ayah, Tafsir, Fiqh, Shariah."
)


@dataclass
class PipelineConfig:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    output_path: str = "transcript.txt"
    flagged_path: str = "flagged.txt"
    quran_corpus: str = os.path.join("data", "quran.json")
    hadith_db_path: str = os.path.join("data", "hadith.db")
    no_hadith: bool = False
    sunnah_api_key: str | None = None
    # --- Tuned decoding parameters (review feedback) ---
    # beam_size=12 gives wider Arabic token exploration on T4
    beam_size: int = 12
    # best_of=8 samples 8 decodings for multilingual stability
    best_of: int = 8
    # patience=1.8 allows longer beam exploration before pruning
    patience: float = 1.8
    # Tighter temperature fallback — high temps cause hallucinations
    # in Islamic speech. Only (0.0, 0.1, 0.2) to avoid invented Arabic.
    temperature: tuple[float, ...] = (0.0, 0.1, 0.2)
    # no_speech_threshold=0.52 — 0.45 is too aggressive and clips
    # soft Quran recitation, dua, and nasheed. 0.52 is safer.
    no_speech_threshold: float = 0.52
    # Urdu-primary lectures: lock transcription language to reduce translation drift.
    primary_language: str = "ur"
    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    profile: str = "default"
    # Second-pass Arabic re-decode for Quran quotes / formal Arabic openings.
    enable_dual_pass_arabic: bool = True
    dual_pass_max_window_s: float = 45.0
    dual_pass_max_gap_s: float = 2.0
    # Chunked transcription with sliding contextual prompts (long lectures).
    contextual_prompt_chunk_s: float = 120.0
    contextual_prompt_min_duration_s: float = 180.0
    # Merge Urdu segments that contain long Arabic quotes before matching.
    merge_arabic_quote_blocks: bool = True
    hadith_workers: int = 4
    low_confidence_threshold: float = 0.35
    low_quality_threshold: float = 0.25
    # --- Hallucination suppression (review-tuned) ---
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 4
    # compression_ratio_threshold=1.8 — 1.6 over-filters Quran which has
    # naturally repetitive morphology. Pipeline uses adaptive threshold:
    # 1.8 for normal, relaxed to 2.2 for detected Arabic/Quran segments.
    compression_ratio_threshold: float = 1.8
    log_prob_threshold: float = -0.7
    hallucination_silence_threshold: float = 1.0
    # --- VAD tuning ---
    # min_silence=250ms for better ayah continuity (500ms cuts pauses)
    vad_min_silence_ms: int = 250
    vad_speech_pad_ms: int = 300
    # --- Chunk overlap ---
    chunk_length: int = 30
    chunk_overlap: int = 2
    # --- Adaptive condition_on_previous_text ---
    # NOT globally enabled. Pipeline uses adaptive logic:
    # enables only when prev segment confidence > 0.85,
    # no hallucination flags, same language, silence < 3s.
    condition_on_previous_text: bool = False
    # --- Audio preprocessing ---
    preprocess_audio: bool = True
    denoise_audio: bool = False
    # --- Language-locked decoding ---
    # When segment language confidence > 0.90, lock to that language.
    # Per-segment only — never globally for the entire file.
    enable_language_locking: bool = True
    language_lock_threshold: float = 0.90
    allow_quranic_repetition: bool = False
    # --- Diagnostics ---
    enable_diagnostics: bool = False
    diagnostics_path: str = "diagnostics.jsonl"
    # --- Retry decoding ---
    # If a segment has high compression / low confidence, retry with
    # safer settings (higher beam, condition_on_previous_text=False).
    enable_retry_decoding: bool = True
    retry_logprob_threshold: float = -0.9
    retry_beam_boost: int = 4
    # --- Hadith matcher (local DB) ---
    hadith_token_set_threshold: int = 78
    hadith_combined_threshold: int = 78
    # --- Hadith verification (retrieval-assisted, Stage 1: suggestion only) ---
    # When enabled, matched Hadith segments are verified against retrieval
    # results with token-level divergence analysis and acoustic gating.
    # Stage 1 generates suggestions only — no transcript mutation.
    enable_hadith_verification: bool = False
    # --- Resource limits ---
    max_file_size_mb: float = 500.0
    max_audio_duration_s: float = 14400.0
    # --- Logging ---
    verbose: bool = False

    @classmethod
    def from_env_defaults(cls) -> PipelineConfig:
        # Colab-friendly env overrides; CLI still has highest precedence.
        device = os.environ.get("STT_DEVICE", "cuda")
        compute_type = os.environ.get("STT_COMPUTE_TYPE", "float16")
        model = os.environ.get("STT_MODEL", "large-v3")
        primary_language = os.environ.get("STT_PRIMARY_LANGUAGE", "ur")
        profile = os.environ.get("STT_PROFILE", "default")
        cfg = cls(
            model=model,
            device=device,
            compute_type=compute_type,
            primary_language=primary_language,
            profile=profile,
            sunnah_api_key=os.environ.get("SUNNAH_API_KEY"),
        )
        return apply_profile(cfg)


def apply_profile(config: PipelineConfig) -> PipelineConfig:
    """Apply Colab-oriented decoding presets on top of base config."""
    profile = (config.profile or "default").lower()
    if profile == "colab-fast":
        return replace(
            config,
            model="large-v3" if config.model == "large-v3" else config.model,
            beam_size=5,
            best_of=3,
            patience=1.2,
            enable_dual_pass_arabic=False,
            enable_retry_decoding=False,
            preprocess_audio=True,
        )
    if profile == "colab-accurate":
        return replace(
            config,
            model="large-v3",
            beam_size=8,
            best_of=5,
            patience=1.5,
            primary_language="ur",
            no_speech_threshold=0.52,
            enable_dual_pass_arabic=True,
            enable_retry_decoding=True,
            preprocess_audio=True,
        )
    if profile == "urdu-bayan":
        return replace(
            config,
            model="large-v3",
            beam_size=8,
            best_of=5,
            patience=1.5,
            primary_language="ur",
            condition_on_previous_text=False,
            compression_ratio_threshold=1.8,
            no_speech_threshold=0.52,
            temperature=(0.0, 0.1, 0.2),
            enable_language_locking=False,
            allow_quranic_repetition=False,
        )
    if profile == "quran-recitation":
        return replace(
            config,
            model="large-v3",
            beam_size=12,
            best_of=8,
            patience=1.8,
            primary_language="ar",
            condition_on_previous_text=False,
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.45,
            temperature=(0.0, 0.1),
            allow_quranic_repetition=True,
        )
    return config
