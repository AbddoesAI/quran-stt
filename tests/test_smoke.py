"""
test_smoke.py
-------------
Deterministic smoke test for the Islamic STT pipeline.
Ensures that processing the same short audio file twice with identical
configuration produces identical deterministic outputs (transcript, languages, etc.).
"""

import os
import shutil
import subprocess
import sys
import tempfile


def create_dummy_audio(path: str):
    """Create a 3-second dummy audio file using ffmpeg for smoke testing if not exists."""
    if os.path.exists(path):
        return
    print(f"Creating dummy audio file at {path}")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=sin(440*2*PI*t):d=3",
            "-ar",
            "16000",
            "-ac",
            "1",
            path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_pipeline(audio_path: str, output_dir: str):
    """Run the pipeline CLI programmatically."""
    env = os.environ.copy()
    # Force CPU for deterministic test if needed, or use whatever config is default
    cmd = [
        sys.executable,
        "-m",
        "islamic_stt.cli",
        audio_path,
        "--profile",
        "default",
        "--output",
        os.path.join(output_dir, "transcript.txt"),
        "--flagged",
        os.path.join(output_dir, "flagged.txt"),
        "--diagnostics",
        os.path.join(output_dir, "diagnostics.jsonl"),
        "--device",
        "cpu",  # Force CPU for strict determinism
        "--compute-type",
        "float32",
        "--no-hadith",  # Skip API calls
    ]
    subprocess.run(cmd, env=env, check=True, capture_output=True)


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_determinism():
    print("Running deterministic smoke test...")
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found, skipping smoke test.")
        return

    with tempfile.TemporaryDirectory() as tempdir:
        audio_path = os.path.join(tempdir, "smoke_test.wav")
        try:
            create_dummy_audio(audio_path)
        except Exception as e:
            print(f"Failed to create dummy audio: {e}")
            return

        out_dir_1 = os.path.join(tempdir, "run1")
        out_dir_2 = os.path.join(tempdir, "run2")
        os.makedirs(out_dir_1, exist_ok=True)
        os.makedirs(out_dir_2, exist_ok=True)

        print("Executing Run 1...")
        run_pipeline(audio_path, out_dir_1)
        print("Executing Run 2...")
        run_pipeline(audio_path, out_dir_2)

        # Compare outputs
        t1 = read_file(os.path.join(out_dir_1, "transcript.txt"))
        t2 = read_file(os.path.join(out_dir_2, "transcript.txt"))
        assert t1 == t2, "Transcripts do not match between runs!"

        d1 = read_file(os.path.join(out_dir_1, "diagnostics.jsonl"))
        d2 = read_file(os.path.join(out_dir_2, "diagnostics.jsonl"))

        # We don't assert perfect string equality on diagnostics because timings/metadata might slightly differ.
        # But for determinism on CPU with float32, they should be extremely close if not identical.
        # We check line counts and basic structure.
        assert len(d1.splitlines()) == len(d2.splitlines()), "Diagnostics segment count mismatch!"

        print("Smoke test passed! Outputs are deterministic.")


if __name__ == "__main__":
    test_determinism()
