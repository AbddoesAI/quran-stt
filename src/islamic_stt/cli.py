"""
cli.py
------
Command-line interface for the Islamic STT pipeline.

Improvements from review plan:
  #20 — Better CLI with verbose flag and progress metrics
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from colorama import Fore, Style, init as colorama_init

from islamic_stt.config import PipelineConfig
from islamic_stt.logging_utils import configure_logging
from islamic_stt.pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    env_defaults = PipelineConfig.from_env_defaults()
    p = argparse.ArgumentParser(
        description="Islamic Lecture STT Pipeline — Transcribe, Detect Quran & Hadith",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio", help="Path to input audio file")
    p.add_argument("--model", default=env_defaults.model,
                   help="Whisper model size")
    p.add_argument("--device", default=env_defaults.device,
                   choices=["cuda", "cpu"],
                   help="Compute device")
    p.add_argument("--compute-type", default=env_defaults.compute_type,
                   choices=["float16", "int8", "int8_float16", "float32"],
                   help="Model precision")
    p.add_argument("--output", default="transcript.txt",
                   help="Output transcript path")
    p.add_argument("--flagged", default="flagged.txt",
                   help="Flagged segments output path")
    p.add_argument("--quran-corpus", default=os.path.join("data", "quran.json"),
                   help="Path to quran.json corpus")
    p.add_argument("--no-hadith", action="store_true",
                   help="Skip Hadith matching (faster)")
    p.add_argument("--no-speech-threshold", type=float, default=0.6,
                   help="No-speech probability threshold")
    p.add_argument("--primary-language", default=env_defaults.primary_language,
                   choices=["ur", "ar", "en", "auto"],
                   help="Primary language of the lecture")
    p.add_argument("--beam-size", type=int, default=8,
                   help="Beam search width (8 recommended for T4)")
    p.add_argument("--best-of", type=int, default=5,
                   help="Number of candidate decodings to sample")
    p.add_argument("--patience", type=float, default=1.5,
                   help="Beam search patience factor")
    p.add_argument("--hadith-workers", type=int, default=4,
                   help="Parallel workers for Hadith API calls")
    p.add_argument("--hadith-db", default=os.path.join("data", "hadith.db"),
                   help="Path to local Hadith SQLite database")
    # --- New flags ---
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug-level logging")
    p.add_argument("--max-file-size", type=float, default=500.0,
                   help="Maximum audio file size in MB")
    p.add_argument("--max-duration", type=float, default=14400.0,
                   help="Maximum audio duration in seconds")
    return p


def main() -> None:
    colorama_init(autoreset=True)

    args = build_arg_parser().parse_args()

    # Configure logging level based on --verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(log_level)

    logger = logging.getLogger(__name__)

    config = PipelineConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        output_path=args.output,
        flagged_path=args.flagged,
        quran_corpus=args.quran_corpus,
        hadith_db_path=args.hadith_db,
        no_hadith=args.no_hadith,
        sunnah_api_key=os.environ.get("SUNNAH_API_KEY"),  # env-only, never CLI
        no_speech_threshold=args.no_speech_threshold,
        primary_language=args.primary_language,
        beam_size=args.beam_size,
        best_of=args.best_of,
        patience=args.patience,
        hadith_workers=args.hadith_workers,
        verbose=args.verbose,
        max_file_size_mb=args.max_file_size,
        max_audio_duration_s=args.max_duration,
    )

    # --- Run pipeline with top-level exception handling (#4) ---
    try:
        stats = run_pipeline(config=config, audio_path=args.audio)
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)

    # --- Print summary (#21 — progress metrics) ---
    print()
    print(Fore.CYAN + "═" * 50)
    print(Fore.CYAN + " Pipeline Complete" + Style.RESET_ALL)
    print(Fore.CYAN + "═" * 50 + Style.RESET_ALL)
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        print(f"  {label}: {v}")
    print(Fore.CYAN + "═" * 50 + Style.RESET_ALL)


if __name__ == "__main__":
    main()
