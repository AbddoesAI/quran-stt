# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-09

### Added
- `src/` layout with proper Python packaging (`pyproject.toml`)
- GPU auto-detection with graceful CPU fallback
- Input validation (file size, type, extension checks)
- Resource limits (`--max-file-size`, `--max-duration`)
- Unified segment quality scoring (logprob + word confidence + speech prob)
- Rotating file logging (`logs/islamic_stt.log`, 5 MB × 3 backups)
- Orphan word absorption in segment merger
- `HadithMatcher` context manager protocol for session cleanup
- `.gitignore`, `.env.example`, `LICENSE`, `CONTRIBUTING.md`
- `pyproject.toml` with `[project.scripts]` entry point
- GitHub Actions CI workflow
- Proper pytest test structure

### Fixed
- **CRITICAL**: `import json` was at bottom of `pipeline.py` — `JSONDecodeError` catch was broken
- **CRITICAL**: `run_pipeline()` mutated caller's `PipelineConfig` dataclass
- **SECURITY**: API key was exposed in CLI `--help` output and `ps aux`
- `hadith_match: Optional[object]` broke `asdict()` — now properly typed as `HadithMatch`
- `_run_lingua()` crashed with `NameError` when lingua not installed
- Duplicate hallucination blocklists consolidated into `arabic_utils.py`

### Changed
- All `print()` calls replaced with structured `logging`
- Matcher Protocol upgraded with `@runtime_checkable` and `Optional[Any]` return
- `_FlaggedEntry` frozen dataclass replaces untyped `dict` in `FlaggedHandler`
- Private `_normalise_arabic` removed from `__all__` exports

## [0.1.0] — 2026-05-08

### Added
- Initial working pipeline: transcription → matching → output
- Whisper large-v3 integration via faster-whisper
- Local Quran corpus matching (exact + fuzzy via rapidfuzz)
- Sunnah.com API Hadith matching with caching
- Layered language detection (Whisper → langdetect → lingua)
- Arabic text normalization (tashkeel, alef, hamza, tatweel)
- Segment merging for readable transcripts
- Multi-format output (TXT, JSON, SRT)
- Flagged segment collection for human review
