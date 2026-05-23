"""
setup_corpus.py
---------------
One-time setup script that downloads the quran-json corpus and assembles
it into the single data/quran.json file expected by quran_matcher.py.

Source : https://github.com/risan/quran-json
Format : fetches the combined 'quran.json' from the jsDelivr CDN.

Run once before using the pipeline:
    python setup_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import requests

__all__ = ["download_corpus"]

# jsDelivr CDN mirror — no GitHub auth required, works in Colab.
CORPUS_URL = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran.json"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = str(Path(DATA_DIR).resolve())  # Fix 11: sanitize path
OUTPUT_PATH = os.path.join(DATA_DIR, "quran.json")

# SHA-256 of quran-json@3.1.2 from jsDelivr CDN.
# Generated with: python -c "import hashlib; print(hashlib.sha256(open('data/quran.json','rb').read()).hexdigest())"
# Set to None only if using a custom corpus that differs from the published version.
EXPECTED_SHA256: str | None = "b6d8a68a9278e15b58cf96fec5f35a9a6ba1ee83a8a23a00e7e3c03e7c0bf2d7"


def download_corpus() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isfile(OUTPUT_PATH):
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f"Corpus already present at {OUTPUT_PATH} ({size_kb:.0f} KB). Skipping download.")
        _validate(OUTPUT_PATH)
        return

    print(f"Downloading Quran corpus from:\n  {CORPUS_URL}\n")

    response = requests.get(CORPUS_URL, timeout=60, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunks = []

    for chunk in response.iter_content(chunk_size=8192):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            sys.stdout.write(f"\r  {downloaded / 1024:.0f} / {total / 1024:.0f} KB  ({pct:.1f}%)")
            sys.stdout.flush()

    print()

    raw = b"".join(chunks)

    # Integrity check (MED-7 fix)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if EXPECTED_SHA256 is not None and actual_sha != EXPECTED_SHA256:
        print(
            f"\n⚠  INTEGRITY CHECK FAILED\n"
            f"  Expected SHA-256: {EXPECTED_SHA256}\n"
            f"  Got:             {actual_sha}\n"
            f"  The downloaded file may be corrupted or tampered with.\n"
            f"  Delete {OUTPUT_PATH} and re-run, or update EXPECTED_SHA256."
        )
        sys.exit(1)

    with open(OUTPUT_PATH, "wb") as fh:
        fh.write(raw)

    size_kb = len(raw) / 1024
    print(f"\nSaved to {OUTPUT_PATH}  ({size_kb:.0f} KB)")
    print(f"  SHA-256: {actual_sha}")
    if EXPECTED_SHA256 is None:
        print(
            f'  TIP: Set EXPECTED_SHA256 = "{actual_sha}" in setup_corpus.py '
            f"to enable integrity checks on future downloads."
        )
    _validate(OUTPUT_PATH)


def _validate(path: str) -> None:
    """Quick sanity check: confirm the JSON has the expected structure."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    assert isinstance(data, list), "Expected top-level list of surahs"
    verse_count = sum(len(s.get("verses", [])) for s in data)
    surah_count = len(data)

    print(
        f"Validation OK — {surah_count} surahs, {verse_count} ayahs loaded.\n"
        f"(Expected: 114 surahs, 6236 ayahs)"
    )

    if surah_count != 114:
        print(f"⚠  WARNING: expected 114 surahs but got {surah_count}.")
    if verse_count != 6236:
        print(f"⚠  WARNING: expected 6236 ayahs but got {verse_count}.")


if __name__ == "__main__":
    download_corpus()
