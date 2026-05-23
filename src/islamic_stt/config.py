from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    # --- Tuned decoding parameters for T4 GPU ---
    # beam_size=8 gives wider exploration for code-switching (T4 has 16GB)
    beam_size: int = 8
    # best_of=5 samples 5 decodings and picks the best (major accuracy gain)
    best_of: int = 5
    # patience=1.5 allows beams to explore longer before pruning
    patience: float = 1.5
    # Temperature fallback: if decoding fails at 0, retry at higher temps
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)
    no_speech_threshold: float = 0.6
    # language=None enables auto-detection; the strong initial_prompt
    # prevents translation. This is critical for Arabic accuracy.
    primary_language: str = "ur"
    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    hadith_workers: int = 4
    low_confidence_threshold: float = 0.35
    low_quality_threshold: float = 0.25
    # --- Tuned hallucination suppression ---
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    compression_ratio_threshold: float = 2.0
    log_prob_threshold: float = -0.8
    hallucination_silence_threshold: float = 1.0
    # --- VAD tuning ---
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 300
    # --- Resource limits ---
    max_file_size_mb: float = 500.0
    max_audio_duration_s: float = 14400.0
    # --- Logging ---
    verbose: bool = False

    @classmethod
    def from_env_defaults(cls) -> "PipelineConfig":
        # Colab-friendly env overrides; CLI still has highest precedence.
        device = os.environ.get("STT_DEVICE", "cuda")
        compute_type = os.environ.get("STT_COMPUTE_TYPE", "float16")
        model = os.environ.get("STT_MODEL", "large-v3")
        primary_language = os.environ.get("STT_PRIMARY_LANGUAGE", "ur")
        return cls(
            model=model,
            device=device,
            compute_type=compute_type,
            primary_language=primary_language,
            sunnah_api_key=os.environ.get("SUNNAH_API_KEY"),
        )
