# Contributing to Quran STT

Thank you for your interest in contributing! This project aims to build a
production-grade Islamic lecture transcription pipeline.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<you>/quran-stt.git`
3. Create a virtual environment: `python -m venv .venv && .venv\Scripts\activate`
4. Install in dev mode: `pip install -e ".[dev]"`
5. Download the corpus: `python scripts/setup_corpus.py`
6. Run tests: `pytest`

## Development Guidelines

### Code Style
- **Python 3.10+** — use modern type hints (`str | None` not `Optional[str]`)
- **`from __future__ import annotations`** at the top of every module
- **Logging** — use `logging.getLogger(__name__)`, never bare `print()`
- **Dataclasses** — prefer over plain dicts for structured data

### Architecture Rules
- **Never mutate `PipelineConfig`** — use local variables for overrides
- **`arabic_utils.py` is the single source of truth** for normalization and blocklists
- **Never pass API keys via CLI** — environment variables only
- **Respect language boundaries** — never merge segments across detected languages

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `docs:` documentation only
- `test:` test additions/changes
- `chore:` maintenance tasks

### Testing
- Add tests for new features in `tests/`
- Run `pytest` before submitting a PR
- Ensure `py_compile` passes on all modules

## Reporting Issues

Please include:
- Python version and OS
- GPU model and VRAM (if relevant)
- Steps to reproduce
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
