"""
cli.py
------
Command-line interface for the Islamic STT pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from colorama import Fore, Style
from colorama import init as colorama_init

from islamic_stt.config import PipelineConfig, apply_profile
from islamic_stt.logging_utils import configure_logging
from islamic_stt.pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = apply_profile(PipelineConfig.from_env_defaults())
    p = argparse.ArgumentParser(
        description="Islamic Lecture STT Pipeline — Transcribe, Detect Quran & Hadith",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio", help="Path to input audio file")
    p.add_argument(
        "--profile",
        default=defaults.profile,
        choices=["default", "colab-fast", "colab-accurate", "urdu-bayan", "quran-recitation"],
        help="Preset tuning profile (urdu-bayan or quran-recitation recommended for production)",
    )
    p.add_argument("--model", default=defaults.model, help="Whisper model size")
    p.add_argument(
        "--device",
        default=defaults.device,
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    p.add_argument(
        "--compute-type",
        default=defaults.compute_type,
        choices=["float16", "int8", "int8_float16", "float32"],
        help="Model precision",
    )
    p.add_argument("--output", default=defaults.output_path, help="Output transcript path")
    p.add_argument("--flagged", default=defaults.flagged_path, help="Flagged segments path")
    p.add_argument(
        "--quran-corpus",
        default=defaults.quran_corpus,
        help="Path to quran.json corpus",
    )
    p.add_argument("--no-hadith", action="store_true", help="Skip Hadith matching (faster)")
    p.add_argument(
        "--no-speech-threshold",
        type=float,
        default=defaults.no_speech_threshold,
        help="No-speech probability threshold",
    )
    p.add_argument(
        "--primary-language",
        default=defaults.primary_language,
        choices=["ur", "ar", "en", "auto"],
        help="Primary language of the lecture",
    )
    p.add_argument("--beam-size", type=int, default=defaults.beam_size, help="Beam search width")
    p.add_argument("--best-of", type=int, default=defaults.best_of, help="Decoder samples")
    p.add_argument("--patience", type=float, default=defaults.patience, help="Beam patience")
    p.add_argument(
        "--hadith-workers",
        type=int,
        default=defaults.hadith_workers,
        help="Parallel workers for Hadith API calls",
    )
    p.add_argument(
        "--hadith-db",
        default=defaults.hadith_db_path,
        help="Path to local Hadith SQLite database",
    )
    p.add_argument(
        "--no-dual-pass",
        action="store_true",
        help="Disable second-pass Arabic re-decode",
    )
    p.add_argument(
        "--diagnostics",
        default=None,
        help="Write per-segment diagnostics JSONL to this path",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug-level logging")
    p.add_argument(
        "--max-file-size",
        type=float,
        default=defaults.max_file_size_mb,
        help="Maximum audio file size in MB",
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=defaults.max_audio_duration_s,
        help="Maximum audio duration in seconds",
    )
    return p


def main() -> None:
    colorama_init(autoreset=True)
    args = build_arg_parser().parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    config = apply_profile(
        PipelineConfig(
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            output_path=args.output,
            flagged_path=args.flagged,
            quran_corpus=args.quran_corpus,
            hadith_db_path=args.hadith_db,
            no_hadith=args.no_hadith,
            sunnah_api_key=os.environ.get("SUNNAH_API_KEY"),
            no_speech_threshold=args.no_speech_threshold,
            primary_language=args.primary_language,
            beam_size=args.beam_size,
            best_of=args.best_of,
            patience=args.patience,
            hadith_workers=args.hadith_workers,
            profile=args.profile,
            enable_dual_pass_arabic=not args.no_dual_pass,
            enable_diagnostics=args.diagnostics is not None,
            diagnostics_path=args.diagnostics or "diagnostics.jsonl",
            verbose=args.verbose,
            max_file_size_mb=args.max_file_size,
            max_audio_duration_s=args.max_duration,
        )
    )

    logger.info(
        "Profile=%s | lang=%s | beam=%d | dual_pass=%s",
        config.profile,
        config.primary_language,
        config.beam_size,
        config.enable_dual_pass_arabic,
    )

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

    print()
    print(Fore.CYAN + "═" * 50)
    print(Fore.CYAN + " Pipeline Complete" + Style.RESET_ALL)
    print(Fore.CYAN + "═" * 50 + Style.RESET_ALL)
    for k, v in stats.items():
        print(f"  {k.replace('_', ' ').title()}: {v}")
    print(Fore.CYAN + "═" * 50 + Style.RESET_ALL)


if __name__ == "__main__":
    main()
