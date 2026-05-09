# 🕌 Quran STT

> AI-powered Islamic lecture transcription with automatic Quran & Hadith detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **local, GPU-accelerated** speech-to-text pipeline that transcribes Islamic
lectures delivered in **Urdu, English, and Arabic**. Beyond raw transcription,
it automatically detects Quranic quotations, matches Hadith references, and
flags uncertain segments for human review.

---

## ✨ Features

- **Multilingual Transcription** — Urdu, Arabic, and English via Whisper large-v3
- **Quran Detection** — Matches recited ayahs against a local 6,236-ayah corpus
- **Hadith Detection** — Verifies Hadith quotes via the Sunnah.com API
- **Islamic Formula Recognition** — Identifies Salawat, Takbir, Tasbih, and 30+ formulas
- **Smart Language Detection** — Layered Whisper → langdetect → lingua pipeline
- **Segment Merging** — Combines Whisper fragments into readable paragraphs
- **Multi-Format Output** — Generates TXT, JSON, and SRT files
- **Human Review** — Flagged segments with word-level confidence scores
- **GPU Auto-Detection** — CUDA float16 with graceful CPU fallback

---

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU with ≥ 10 GB VRAM (recommended) or CPU (slower)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (for GPU acceleration)

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

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Standard install
pip install -e .

# With development tools (pytest, ruff)
pip install -e ".[dev]"
```

### 4. Install PyTorch with CUDA

```bash
# For CUDA 12.1 (adjust for your CUDA version)
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 5. Download the Quran corpus

```bash
python scripts/setup_corpus.py
```

### 6. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your Sunnah.com API key
```

Get a free API key at [sunnah.com/developers](https://sunnah.com/developers).

---

## 💻 Usage

### Command Line

```bash
# Basic usage (GPU auto-detected)
quran-stt lecture.mp3

# Specify output path
quran-stt lecture.mp3 --output output/transcript.txt

# CPU mode
quran-stt lecture.mp3 --device cpu --compute-type int8

# Skip Hadith matching (faster, no API needed)
quran-stt lecture.mp3 --no-hadith

# Verbose logging
quran-stt lecture.mp3 -v
```

### Python API

```python
from islamic_stt.config import PipelineConfig
from islamic_stt.pipeline import run_pipeline

config = PipelineConfig(
    model="large-v3",
    device="cuda",
    primary_language="ur",
    no_hadith=True,  # skip API calls
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
├── docs/                      # Documentation
│   └── context.md             # Architecture reference
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

## ⚙️ Configuration

| CLI Flag | Default | Description |
|---|---|---|
| `--model` | `large-v3` | Whisper model size |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--compute-type` | `float16` | Model precision |
| `--primary-language` | `ur` | Primary lecture language |
| `--beam-size` | `5` | Beam search width |
| `--no-hadith` | `false` | Skip Hadith API matching |
| `--no-speech-threshold` | `0.6` | Silence detection threshold |
| `--max-file-size` | `500` | Max audio file size (MB) |
| `--max-duration` | `14400` | Max audio duration (seconds) |
| `-v, --verbose` | `false` | Debug-level logging |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test
pytest tests/test_arabic_utils.py
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 Whisper backend
- [quran-json](https://github.com/risan/quran-json) — Quran corpus dataset
- [Sunnah.com](https://sunnah.com) — Hadith API
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) — Fast string matching
