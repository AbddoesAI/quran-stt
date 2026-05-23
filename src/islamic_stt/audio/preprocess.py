"""
preprocess.py
-------------
Audio preprocessing pipeline for Islamic STT.

Performs:
  1. Resample to 16kHz mono s16 PCM
  2. Highpass (120Hz) + lowpass (7600Hz) to remove rumble/hiss
  3. Loudness normalization (EBU R128: -16 LUFS)
  4. Optional DeepFilterNet denoising

Returns a path to the cleaned temporary WAV file.
The caller is responsible for deleting the temp file after use.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["preprocess_audio"]


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def preprocess_audio(
    input_path: str,
    *,
    output_dir: str | None = None,
    denoise: bool = False,
) -> str:
    """
    Preprocess audio for optimal Whisper performance.

    Parameters
    ----------
    input_path  : Path to the raw audio file.
    output_dir  : Directory for the cleaned file (default: system temp).
    denoise     : If True, attempt DeepFilterNet denoising after FFmpeg.

    Returns
    -------
    Path to the cleaned WAV file (16kHz, mono, s16 PCM).
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if not _ffmpeg_available():
        logger.warning(
            "FFmpeg not found — skipping preprocessing. Install FFmpeg for +8-18%% WER improvement."
        )
        return input_path

    # Create output path
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="islamic_stt_")
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(input_path).stem
    cleaned_path = os.path.join(output_dir, f"{stem}_cleaned.wav")

    # FFmpeg preprocessing:
    #   -ac 1           : mono
    #   -ar 16000       : 16kHz (Whisper native)
    #   -sample_fmt s16 : signed 16-bit PCM (critical for Whisper)
    #   highpass=f=120  : remove rumble/HVAC noise
    #   lowpass=f=7600  : remove hiss above speech range
    #   loudnorm        : EBU R128 loudness normalization
    #     I=-16         : target loudness
    #     LRA=11        : loudness range
    #     TP=-1.5       : true peak limit (prevents clipping)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-af",
        "highpass=f=120,lowpass=f=7600,loudnorm=I=-16:LRA=11:TP=-1.5",
        cleaned_path,
    ]

    logger.info(
        "Preprocessing: %s → %s", os.path.basename(input_path), os.path.basename(cleaned_path)
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning("FFmpeg preprocessing failed:\n%s", result.stderr[-500:])
            return input_path

        cleaned_size = os.path.getsize(cleaned_path) / (1024 * 1024)
        logger.info("Preprocessed: %.1f MB → %s", cleaned_size, os.path.basename(cleaned_path))

    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timed out — using original audio.")
        return input_path
    except Exception as exc:
        logger.warning("Preprocessing error: %s — using original audio.", exc)
        return input_path

    # Optional DeepFilterNet denoising
    if denoise:
        cleaned_path = _try_deepfilter(cleaned_path) or cleaned_path

    return cleaned_path


def _try_deepfilter(wav_path: str) -> str | None:
    """Attempt DeepFilterNet denoising. Returns denoised path or None."""
    try:
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df()
        # DeepFilterNet processes in-place or to output
        denoised_path = wav_path.replace("_cleaned.wav", "_denoised.wav")
        # Use subprocess if the Python API is not available
        result = subprocess.run(
            ["deepFilter", wav_path, "-o", denoised_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.isfile(denoised_path):
            logger.info("DeepFilterNet denoising applied.")
            return denoised_path
    except ImportError:
        logger.debug("DeepFilterNet not installed — skipping denoising.")
    except Exception as exc:
        logger.debug("DeepFilterNet error: %s", exc)
    return None
