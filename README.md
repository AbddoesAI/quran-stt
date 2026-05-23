# 🕌 Quran STT

> AI-powered Islamic lecture transcription with automatic Quran & Hadith detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **local, GPU-accelerated** speech-to-text pipeline that transcribes Islamic lectures delivered in **Urdu, English, and Arabic**. Beyond raw transcription, it automatically detects Quranic quotations, matches Hadith references, and flags uncertain segments for human review.

This is an accuracy-focused research and engineering project built to solve the unique challenges of code-switched Islamic speech.

---

## ✨ Features

- **Multilingual Transcription**: Native support for Urdu, Arabic, and English via Whisper `large-v3`.
- **Dual-Pass Arabic Decoding**: Special recovery passes for Arabic speech embedded within Urdu lectures.
- **Quran Detection**: Fuzzy matches recited ayahs against a local 6,236-ayah corpus using an Aho-Corasick + Trigram pipeline.
- **Hadith Detection**: Verifies Hadith quotes via local SQLite FTS5 database (34k+ hadiths) with a fallback to the Sunnah.com API.
- **Hallucination Reduction**: Multi-layered filtering (deduplication, phrase blocklists, structural checks) to suppress Whisper's common hallucinations.
- **Smart Language Detection**: A layered Whisper → langdetect → lingua pipeline to detect code-switching boundaries.
- **Segment Merging**: Combines short, fragmented Whisper segments into readable paragraphs without losing word-level timestamps.
- **Confidence Calibration**: Word-level and segment-level confidence scoring for targeted human review.
- **Multi-Format Output**: Generates TXT, JSON (with rich metadata), and SRT files.

---

## 🏗️ Architecture

The pipeline follows a strict, safety-first sequential architecture:

1. **Pre-process**: Audio is normalized and denoised (16kHz, mono, EBU R128).
2. **Transcribe**: `faster-whisper` decodes audio chunks, applying domain-specific initial prompts and repetition penalties.
3. **Classify**: Each segment's language is detected to handle mid-sentence code-switching (Urdu ↔ Arabic).
4. **Merge**: Short fragments are semantically merged based on language and temporal proximity.
5. **Post-process**: Arabic and Urdu texts are structurally repaired (abbreviation expansion, mixed-script sanitization).
6. **Verify**: Arabic segments are routed to Quran/Hadith retrieval matchers. **Audio remains the source of truth**; retrieval never mutates the underlying transcript.
7. **Output**: Transcripts and rich JSON diagnostics are generated.

---

## 🌍 Supported Languages

- **Urdu** (Primary focus: Islamic lectures / Bayans)
- **Arabic** (Quran recitation, Hadith narration, formal speech)
- **English** (Islamic lectures)

*Note: The system handles code-switched speech (e.g., Urdu speakers quoting Arabic) gracefully through dynamic contextual prompting and language-locked re-decoding.*

---

## 📖 Quran & Hadith Verification

Quran and Hadith text verification uses a hybrid approach:
- **Quran**: Uses a local JSON corpus. Matches are scored via partial and token-set ratios, with adaptive thresholding based on acoustic confidence.
- **Hadith**: Uses a local SQLite FTS5 database covering major collections (Bukhari, Muslim, Tirmidhi, Abu Dawud, Nasai, Ibn Majah).
- **Safety First**: Retrieval systems act as a **verification layer, not a generation layer**. They provide citations and confidence scores but do not rewrite the ASR output.

---

## 🛡️ Hallucination Reduction

Whisper is prone to hallucinating religious text, especially when prompted with Islamic terminology. This project implements:
1. **Deduplication Windows**: Semantic overlap merging across chunk boundaries.
2. **Blocklists**: Hardcoded rejection of known Whisper failure loops (e.g., repeating "سبحان الله").
3. **Acoustic Gating**: Segments with high compression ratios or low log-probabilities trigger safer retry-decoding passes.

---

## 🎛️ Profiles

The system includes tuned decoding profiles for different workloads:
- `default`: Balanced accuracy and speed.
- `colab-fast`: Optimized for free-tier T4 GPUs (smaller beams, no dual-pass).
- `colab-accurate`: Maximum quality, full dual-pass recovery.
- `urdu-bayan`: Tuned specifically for Urdu lectures with Arabic quotes.
- `quran-recitation`: High-patience decoding for continuous Tajweed.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/AbddoesAI/quran-stt.git
cd quran-stt
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# or
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
# Standard install
pip install -e .

# With development and evaluation tools
pip install -e ".[dev,perf]"
```

### 4. Install PyTorch with CUDA

```bash
# For CUDA 12.1 (adjust for your specific CUDA version)
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 5. Setup Data & Databases

```bash
# Download the Quran corpus
python scripts/setup_corpus.py

# Build the local Hadith FTS5 Database (Requires data/all_hadiths_clean.csv)
python scripts/build_hadith_db.py
```

---

## ☁️ Colab Setup

You can run this pipeline in Google Colab using a T4 GPU:

```python
!git clone https://github.com/AbddoesAI/quran-stt.git
%cd quran-stt
!pip install -e .
!python scripts/setup_corpus.py
# Run the pipeline
!quran-stt /content/lecture.mp3 --profile colab-accurate
```

---

## 💻 Usage Examples

### Command Line Interface

```bash
# Basic usage (GPU auto-detected)
quran-stt lecture.mp3

# Specify profile and output path
quran-stt lecture.mp3 --profile urdu-bayan --output output/transcript.txt

# CPU mode with int8 quantization
quran-stt lecture.mp3 --device cpu --compute-type int8

# Generate detailed diagnostic logs (useful for debugging hallucinations)
quran-stt lecture.mp3 --diagnostics diagnostics.jsonl
```

### Python API

```python
from islamic_stt.config import PipelineConfig
from islamic_stt.pipeline import run_pipeline

config = PipelineConfig(
    model="large-v3",
    device="cuda",
    primary_language="ur",
    profile="urdu-bayan"
)

stats = run_pipeline(config, audio_path="lecture.mp3")
print(f"Transcribed {stats['total_segments']} segments in {stats['elapsed_seconds']}s")
```

### Google Colab

```python
# Cell 1: Install
!pip install faster-whisper langdetect lingua-language-detector rapidfuzz tqdm colorama
!git clone https://github.com/AbddoesAI/quran-stt.git
%cd quran-stt
!pip install -e .
!python scripts/setup_corpus.py

# Cell 2: Transcribe
!quran-stt /content/lecture.mp3 --output /content/transcript.txt
```

---

## 📁 Project Structure

```
quran-stt/
├── src/islamic_stt/           # Main Python package
│   ├── __init__.py            # Package exports
│   ├── __main__.py            # python -m islamic_stt
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # PipelineConfig dataclass
│   ├── pipeline.py            # Main orchestrator
│   ├── logging_utils.py       # Logging configuration
│   ├── core/                  # Transcription & NLP
│   │   ├── arabic_utils.py    # Arabic text normalization
│   │   ├── transcriber.py     # Whisper inference wrapper
│   │   ├── segment_merger.py  # Fragment → paragraph merging
│   │   └── language_detector.py
│   ├── matchers/              # Source matching engines
│   │   ├── base.py            # Matcher protocol
│   │   ├── quran_matcher.py   # Local Quran corpus matching
│   │   └── hadith_matcher.py  # Sunnah.com API matching
│   └── output/                # Output serialization
│       ├── output_handler.py  # TXT/JSON/SRT generation
│       └── flagged_handler.py # Unverified segment collection
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
│   └── setup_corpus.py        # Quran corpus downloader
├── data/                      # Corpus data (gitignored)
├── pyproject.toml             # Project metadata & tool config
├── requirements.txt           # pip dependencies
└── .env.example               # Environment variable template
```

---

## 📊 Output Example

```
════════════════════════════════════════════════════════════════════════════
 ISLAMIC LECTURE TRANSCRIPT
 Duration: 01:23:45  |  Segments: 342
 Languages: AR: 45  EN: 12  UR: 285
════════════════════════════════════════════════════════════════════════════

[00:00:03] [UR] بسم اللہ الرحمن الرحیم، آج ہم بات کریں گے
[00:00:15] [AR] بسم الله الرحمن الرحيم
           ↳ 📖 Quran 1:1 — Al-Fatihah ✓exact (100%)
[00:00:22] [AR] إنما الأعمال بالنيات
           ↳ 📜 Hadith — Bukhari #1 (89%)
[00:01:05] [EN] So the Prophet, peace be upon him, said...
```

---

## 📈 Accuracy Goals & Current Limitations

### Realistic Accuracy Goals
- **Urdu Lectures**: 82–90% usable accuracy
- **Mixed Urdu/Arabic**: 75–88% usable accuracy
- **Quran Recitation**: 70–85% usable accuracy

*(Note: 99% accuracy is not achievable with generic Whisper models without diarization and perfect diacritization recovery. This system aims for the highest realistic ceiling via pipeline engineering).*

### Current Limitations
1. **Diacritization**: Whisper frequently outputs undiacritized Arabic. The normalization pipeline handles this for matching, but transcripts may lack *tashkeel*.
2. **Language Boundaries**: Extremely short Arabic quotes (1-2 words) embedded in Urdu may be misclassified and miss the Quran matching layer.
3. **Retrieval Safety**: Fuzzy matching may occasionally attribute a heavily ASR-corrupted verse to a similar, but incorrect, citation.

---

## 🗺️ Fine-Tuning Roadmap

Fine-tuning is **not** currently recommended until the baseline evaluation harness is fully populated. Future plans include:
1. LoRA fine-tuning of the decoder to inject domain-specific Islamic terminology.
2. Continued pre-training on a large corpus of unlabeled Islamic audio.

---

## ⚖️ Safety Philosophy

* **Audio is the Source of Truth**: The system will never replace spoken words with retrieved database text. Database retrieval is strictly used for *citation* and *verification*.
* **Brutal Technical Honesty**: We do not exaggerate accuracy claims. Hallucinations and errors are expected, which is why confidence scoring and flagging systems are built-in.

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
