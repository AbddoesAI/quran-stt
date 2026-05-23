"""
build_quran_json.py
-------------------
Builds data/quran.json from data/quran-simple.txt for offline usage.

quran-simple.txt format: surah_id|ayah_id|arabic_text
Output: JSON list of 114 surahs with {id, name, verses: [{id, text}]}

This allows the pipeline to run without downloading from the CDN.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
INPUT_PATH = os.path.join(DATA_DIR, "quran-simple.txt")
OUTPUT_PATH = os.path.join(DATA_DIR, "quran.json")

# Surah names (1-114)
SURAH_NAMES = [
    "",
    "Al-Fatihah",
    "Al-Baqarah",
    "Aal-E-Imran",
    "An-Nisa",
    "Al-Ma'idah",
    "Al-An'am",
    "Al-A'raf",
    "Al-Anfal",
    "At-Tawbah",
    "Yunus",
    "Hud",
    "Yusuf",
    "Ar-Ra'd",
    "Ibrahim",
    "Al-Hijr",
    "An-Nahl",
    "Al-Isra",
    "Al-Kahf",
    "Maryam",
    "Taha",
    "Al-Anbiya",
    "Al-Hajj",
    "Al-Mu'minun",
    "An-Nur",
    "Al-Furqan",
    "Ash-Shu'ara",
    "An-Naml",
    "Al-Qasas",
    "Al-Ankabut",
    "Ar-Rum",
    "Luqman",
    "As-Sajdah",
    "Al-Ahzab",
    "Saba",
    "Fatir",
    "Ya-Sin",
    "As-Saffat",
    "Sad",
    "Az-Zumar",
    "Ghafir",
    "Fussilat",
    "Ash-Shura",
    "Az-Zukhruf",
    "Ad-Dukhan",
    "Al-Jathiyah",
    "Al-Ahqaf",
    "Muhammad",
    "Al-Fath",
    "Al-Hujurat",
    "Qaf",
    "Adh-Dhariyat",
    "At-Tur",
    "An-Najm",
    "Al-Qamar",
    "Ar-Rahman",
    "Al-Waqi'ah",
    "Al-Hadid",
    "Al-Mujadila",
    "Al-Hashr",
    "Al-Mumtahanah",
    "As-Saf",
    "Al-Jumu'ah",
    "Al-Munafiqun",
    "At-Taghabun",
    "At-Talaq",
    "At-Tahrim",
    "Al-Mulk",
    "Al-Qalam",
    "Al-Haqqah",
    "Al-Ma'arij",
    "Nuh",
    "Al-Jinn",
    "Al-Muzzammil",
    "Al-Muddaththir",
    "Al-Qiyamah",
    "Al-Insan",
    "Al-Mursalat",
    "An-Naba",
    "An-Nazi'at",
    "Abasa",
    "At-Takwir",
    "Al-Infitar",
    "Al-Mutaffifin",
    "Al-Inshiqaq",
    "Al-Buruj",
    "At-Tariq",
    "Al-A'la",
    "Al-Ghashiyah",
    "Al-Fajr",
    "Al-Balad",
    "Ash-Shams",
    "Al-Layl",
    "Ad-Duhaa",
    "Ash-Sharh",
    "At-Tin",
    "Al-Alaq",
    "Al-Qadr",
    "Al-Bayyinah",
    "Az-Zalzalah",
    "Al-Adiyat",
    "Al-Qari'ah",
    "At-Takathur",
    "Al-Asr",
    "Al-Humazah",
    "Al-Fil",
    "Quraysh",
    "Al-Ma'un",
    "Al-Kawthar",
    "Al-Kafirun",
    "An-Nasr",
    "Al-Masad",
    "Al-Ikhlas",
    "Al-Falaq",
    "An-Nas",
]


def build_quran_json() -> None:
    if not os.path.isfile(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Download quran-simple.txt from https://tanzil.net/download/")
        sys.exit(1)

    print(f"Reading {INPUT_PATH}...")

    # Parse pipe-delimited format
    surahs: dict[int, list[dict]] = defaultdict(list)
    with open(INPUT_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            sid, aid, text = int(parts[0]), int(parts[1]), parts[2]
            surahs[sid].append({"id": aid, "text": text})

    # Build JSON structure matching quran-json@3.1.2 format
    result = []
    for sid in sorted(surahs.keys()):
        name = SURAH_NAMES[sid] if sid < len(SURAH_NAMES) else f"Surah {sid}"
        result.append(
            {
                "id": sid,
                "name": name,
                "verses": surahs[sid],
            }
        )

    # Write
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    verse_count = sum(len(s["verses"]) for s in result)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nBuilt {OUTPUT_PATH}")
    print(f"  Surahs: {len(result)}")
    print(f"  Ayahs:  {verse_count}")
    print(f"  Size:   {size_kb:.0f} KB")

    if len(result) != 114:
        print(f"  ⚠ Expected 114 surahs, got {len(result)}")
    if verse_count != 6236:
        print(f"  ⚠ Expected 6236 ayahs, got {verse_count}")


if __name__ == "__main__":
    build_quran_json()
