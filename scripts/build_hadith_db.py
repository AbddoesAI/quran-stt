"""
build_hadith_db.py
------------------
Ingests all_hadiths_clean.csv into a local SQLite database with FTS5 full-text
search for offline Hadith matching.

This replaces the Sunnah.com API dependency (100 req/hr rate limit) with a
local database of 34,000+ hadiths from the six major collections.

Schema
------
- hadiths       : main table with Arabic text, English translation, metadata
- hadith_fts    : FTS5 virtual table over normalized Arabic text for fast search
- hadith_trigrams: pre-built trigram index for fuzzy candidate filtering

Usage
-----
    python scripts/build_hadith_db.py
    # or
    python -m scripts.build_hadith_db

The output database is written to data/hadith.db (~15-25 MB).
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add parent to path so we can import islamic_stt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from islamic_stt.core.arabic_utils import normalise_arabic

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_DIR / "data"
_CSV_PATH = _PROJECT_DIR / "all_hadiths_clean.csv"
_DB_PATH = _DATA_DIR / "hadith.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_SCHEMA = """
-- Main hadith table
CREATE TABLE IF NOT EXISTS hadiths (
    id INTEGER PRIMARY KEY,
    hadith_id INTEGER,
    collection TEXT NOT NULL,
    chapter_no TEXT,
    hadith_no TEXT NOT NULL,
    chapter TEXT,
    chain_indx TEXT,
    text_ar TEXT NOT NULL,
    text_ar_normalized TEXT NOT NULL,
    text_en TEXT,
    UNIQUE(collection, hadith_no)
);

-- FTS5 full-text search over normalized Arabic
CREATE VIRTUAL TABLE IF NOT EXISTS hadith_fts USING fts5(
    text_ar_normalized,
    content='hadiths',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_collection ON hadiths(collection);
CREATE INDEX IF NOT EXISTS idx_hadith_no ON hadiths(hadith_no);
CREATE INDEX IF NOT EXISTS idx_collection_hadith ON hadiths(collection, hadith_no);
"""

_INSERT_SQL = """
INSERT OR IGNORE INTO hadiths
    (hadith_id, collection, chapter_no, hadith_no, chapter, chain_indx,
     text_ar, text_ar_normalized, text_en)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_FTS_REBUILD = """
INSERT INTO hadith_fts(hadith_fts) VALUES('rebuild');
"""

# ---------------------------------------------------------------------------
# Collection name normalization
# ---------------------------------------------------------------------------

_COLLECTION_MAP: dict[str, str] = {
    "sahih bukhari": "bukhari",
    "sahih muslim": "muslim",
    "jami' al-tirmidhi": "tirmidhi",
    "jami al-tirmidhi": "tirmidhi",
    "sunan abi da'ud": "abudawud",
    "sunan abi dawud": "abudawud",
    "sunan an-nasa'i": "nasai",
    "sunan an-nasai": "nasai",
    "sunan ibn majah": "ibnmajah",
}


def _normalize_collection(raw: str) -> str:
    """Normalize collection name to a canonical short form."""
    return _COLLECTION_MAP.get(raw.strip().lower(), raw.strip().lower())


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_database(csv_path: str | Path = _CSV_PATH, db_path: str | Path = _DB_PATH) -> None:
    """
    Read the CSV and build the SQLite + FTS5 database.
    """
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    if not csv_path.is_file():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB to rebuild fresh
    if db_path.exists():
        logger.info("Removing existing database: %s", db_path)
        db_path.unlink()

    logger.info("Building Hadith database from: %s", csv_path)
    logger.info("Output: %s", db_path)

    t0 = time.perf_counter()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

    # Create schema
    conn.executescript(_CREATE_SCHEMA)

    # Read and insert
    inserted = 0
    skipped = 0
    errors = 0

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        logger.info("CSV columns: %s", header)

        # Columns: id, hadith_id, source, chapter_no, hadith_no, chapter, chain_indx, text_ar, text_en
        batch: list[tuple] = []

        for row_num, row in enumerate(reader, start=2):
            try:
                if len(row) < 9:
                    skipped += 1
                    continue

                (csv_id, hadith_id, source, chapter_no, hadith_no,
                 chapter, chain_indx, text_ar, text_en) = row[:9]

                text_ar = text_ar.strip()
                if not text_ar or len(text_ar) < 10:
                    skipped += 1
                    continue

                collection = _normalize_collection(source)
                text_ar_normalized = normalise_arabic(text_ar)

                batch.append((
                    int(hadith_id) if hadith_id.strip() else 0,
                    collection,
                    chapter_no.strip(),
                    hadith_no.strip(),
                    chapter.strip(),
                    chain_indx.strip(),
                    text_ar,
                    text_ar_normalized,
                    text_en.strip(),
                ))

                # Batch insert every 1000 rows
                if len(batch) >= 1000:
                    conn.executemany(_INSERT_SQL, batch)
                    inserted += len(batch)
                    batch.clear()
                    if inserted % 5000 == 0:
                        logger.info("  Inserted %d rows …", inserted)

            except Exception as exc:
                errors += 1
                if errors <= 10:
                    logger.warning("Row %d error: %s", row_num, exc)

        # Insert remaining
        if batch:
            conn.executemany(_INSERT_SQL, batch)
            inserted += len(batch)

    conn.commit()

    # Build FTS index
    logger.info("Building FTS5 index …")
    conn.execute(_FTS_REBUILD)
    conn.commit()

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM hadiths").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM hadith_fts").fetchone()[0]

    # Collection breakdown
    collections = conn.execute(
        "SELECT collection, COUNT(*) FROM hadiths GROUP BY collection ORDER BY COUNT(*) DESC"
    ).fetchall()

    conn.close()

    elapsed = time.perf_counter() - t0
    db_size_mb = db_path.stat().st_size / (1024 * 1024)

    logger.info("=" * 60)
    logger.info("Hadith Database Built Successfully")
    logger.info("=" * 60)
    logger.info("  Rows inserted : %d", count)
    logger.info("  FTS entries   : %d", fts_count)
    logger.info("  Skipped       : %d", skipped)
    logger.info("  Errors        : %d", errors)
    logger.info("  DB size       : %.1f MB", db_size_mb)
    logger.info("  Time          : %.1fs", elapsed)
    logger.info("  Collections:")
    for coll, cnt in collections:
        logger.info("    %-15s %d", coll, cnt)
    logger.info("=" * 60)


if __name__ == "__main__":
    build_database()
