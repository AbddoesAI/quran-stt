from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_INITIAL_PROMPT = (
    # Urdu instruction: transcribe as spoken, do not translate
    "یہ اسلامی بیان ہے جس میں اردو، عربی اور انگریزی تینوں زبانیں استعمال ہوتی ہیں۔ "
    "ہر زبان کو جوں کا توں لکھیں، ترجمہ نہ کریں۔ "
    # Arabic Quranic/Hadith vocabulary (primes the decoder for Arabic passages)
    "بسم الله الرحمن الرحيم۔ صلى الله عليه وسلم۔ سبحان الله۔ الحمد لله۔ "
    "قرآن، حدیث، صحابہ، تفسیر، فقہ، سنت، بخاری، مسلم۔ "
    # English terms commonly heard in Islamic lectures
    "Islamic, Quran, Hadith, Sunnah, Sahabi, Prophet Muhammad."
)


@dataclass
class PipelineConfig:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    output_path: str = "transcript.txt"
    flagged_path: str = "flagged.txt"
    quran_corpus: str = os.path.join("data", "quran.json")
    no_hadith: bool = False
    sunnah_api_key: str | None = None
    no_speech_threshold: float = 0.6
    primary_language: str = "ur"
    beam_size: int = 5
    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    hadith_workers: int = 4
    low_confidence_threshold: float = 0.35
    # --- Resource limits (#27, #28 from improvement plan) ---
    max_file_size_mb: float = 500.0          # reject audio files > 500 MB
    max_audio_duration_s: float = 14400.0    # reject audio > 4 hours
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
