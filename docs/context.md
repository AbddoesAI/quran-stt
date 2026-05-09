# Islamic STT Pipeline

> **Last updated**: 2026-05-09 (post full code review)
> **Purpose**: Persistent project memory for LLMs and developers.
> **Status**: Hardened Prototype — Production Readiness 7.5/10 (Phase 1 + Code Review complete)

---

## Project Purpose

A **local, GPU-accelerated** speech-to-text pipeline designed to transcribe long Islamic lectures delivered in a complex mix of **Urdu, English, and Arabic**. 

Beyond raw transcription, this system acts as an intelligent theological annotator. It automatically detects Arabic quotations, verifies them against a local Quran corpus, matches them against the Sunnah.com API for Hadith, and flags uncertain or unverified Arabic segments for human review. The ultimate goal is to produce a clean, highly readable, and academically useful transcript of religious audio.

---

## Core Goals

1. **Accurate Multilingual Transcription**: Process Urdu as the primary language while seamlessly handling Arabic (Quran/Hadith) and English (code-switching).
2. **Quran Detection**: Identify when the speaker recites a Quranic ayah, match it, and annotate the transcript with exact surah and ayah references.
3. **Hadith Detection**: Identify Hadith quotations and annotate them with their respective collection and number.
4. **Islamic Formula Detection**: Recognize and format common Islamic phrases (e.g., Salawat, Tasbih, Tahmid).
5. **Human-in-the-Loop Review**: Collect all unverified Arabic segments into a structured `flagged.txt` file for human verification.
6. **Readable Output**: Produce structured transcripts with language badges, paragraph merging, and timestamped blocks.

---

## Key Features

- **Segment Merging**: Intelligently combines short Whisper fragments (1-3 words) into coherent sentences while strictly respecting language boundaries.
- **Multilingual Whisper Tuning**: Custom `repetition_penalty`, `compression_ratio_threshold`, and `no_repeat_ngram_size` configured specifically to prevent hallucinations in multilingual audio.
- **Advanced Arabic Normalization**: Strips tashkeel (diacritics), normalizes alef/hamza variants, and strips bi-directional markers to allow robust fuzzy matching of Whisper's inconsistent Arabic output against heavily diacritized classical corpora.
- **Dual Matching Architecture**: O(1) exact matching for speed, followed by `rapidfuzz` token-set ratio matching for partial/fuzzy quotations.
- **Layered Language Detection**: Uses Whisper's built-in detection, `langdetect`, and `lingua` to confidently differentiate between Urdu and Arabic despite both using the Arabic script.

---

## Supported Languages

| Language | Code | Role | Script Notes |
|----------|------|------|--------------|
| **Urdu** | `ur` | Primary | Arabic script + Urdu-exclusive characters (ٹ ڈ ڑ ں ھ ے). Whisper often struggles distinguishing it from Arabic. |
| **Arabic** | `ar` | Quotations | Classical Arabic for Quran, Hadith, and formulas. Heavily normalized during processing. |
| **English** | `en` | Code-switching | Latin script. Transcribed naturally when the speaker code-switches. |

---

## Main Technologies Used

- **faster-whisper (1.1.1)**: Core transcription engine using CTranslate2 backend. Significantly faster and more memory-efficient than OpenAI's reference implementation.
- **Whisper large-v3**: The underlying model. Offers the best multilingual accuracy and zero-shot translation resistance.
- **CUDA float16**: Target compute type for NVIDIA GPUs (T4/A100) to maximize inference speed.
- **rapidfuzz (3.9.7)**: Extremely fast string matching library used for fuzzy Quran and Hadith matching.
- **lingua-language-detector (2.0.2)** & **langdetect (1.0.9)**: Used for per-segment language classification to resolve Arabic/Urdu ambiguity.
- **requests (2.32.3)**: Handles HTTP calls to the Sunnah.com API.
- **Quran Corpus (quran-json@3.1.2)**: Local 6,236-ayah JSON dataset.
- **Sunnah.com API**: Free-tier API used for Hadith verification.

---

## High-Level System Architecture

The system operates as a linear, sequential pipeline with synchronous and asynchronous elements.

1. **Transcription Flow**: Audio is loaded into `faster-whisper`. The model transcribes the audio, applies VAD (Voice Activity Detection), and yields `TranscriptSegment` objects with word-level timestamps.
2. **Cleanup Flow**: Segments pass through hallucination filters and deduplication.
3. **Merge Flow**: The `segment_merger` combines short, fragmented segments into longer, readable blocks while respecting language boundaries and maximum duration caps.
4. **Detection Flow**: Each merged segment is analyzed by the language detector.
5. **Matching Flow**: If a segment is flagged as Arabic (`ar`):
   - It is checked for Islamic formulas.
   - It is checked against the local Quran corpus (exact, then fuzzy).
   - If no match, it is queued for Hadith matching via the Sunnah.com API (processed asynchronously via ThreadPoolExecutor).
6. **Output Flow**: Segments are enriched with match metadata and written to `transcript.txt` and `flagged.txt` alongside JSON and SRT exports.

---

## Current Folder Structure

```text
Quran STT/
├── islamic_stt/               # Core application package
│   ├── __init__.py            # Package export
│   ├── cli.py                 # Command-line interface parser
│   ├── config.py              # PipelineConfig dataclass and environment defaults
│   ├── pipeline.py            # Main orchestrator linking all modules
│   ├── logging_utils.py       # Centralized Python logging configuration
│   └── matchers/              # Matcher architecture (Plugin basis)
│       ├── __init__.py
│       └── base.py            # Base Matcher Protocol
├── transcriber.py             # Whisper inference wrapper and deduplication
├── segment_merger.py          # Merges short fragments into readable blocks
├── language_detector.py       # Resolves Urdu vs Arabic language tags
├── arabic_utils.py            # Text normalization, tashkeel stripping
├── quran_matcher.py           # Quran exact/fuzzy matching & Formula detection
├── hadith_matcher.py          # Sunnah.com API client for Hadith
├── output_handler.py          # Writes final TXT, JSON, and SRT files
├── flagged_handler.py         # Collects and writes unverified Arabic segments
├── setup_corpus.py            # One-time script to download quran.json
├── main.py                    # Top-level entry script
├── requirements.txt           # Python dependencies
├── data/                      # Local data storage
│   └── quran.json             # Downloaded Quran corpus
├── cache/                     # Hadith API shelve cache
└── logs/                      # Rotating log files (auto-created)
    └── islamic_stt.log        # 5 MB × 3 backups
```

---

## Detailed Module Responsibilities

### `main.py` & `islamic_stt/cli.py`
- **Role**: Entry point and CLI parser.
- **Logic**: Parses arguments, initializes `PipelineConfig`, configures logging, and triggers `run_pipeline()`.

### `transcriber.py`
- **Role**: Wraps `faster-whisper`.
- **Inputs**: Audio path, Whisper parameters.
- **Outputs**: `List[TranscriptSegment]`.
- **Critical Logic**: Handles `float16` to `int8` CPU fallback. Contains `_deduplicate_segments` (120s window) to kill Whisper hallucination loops.
- **Limitations**: Memory heavy. Dictates the speed of the entire application.

### `segment_merger.py`
- **Role**: Combines fragmented segments.
- **Inputs**: `List[TranscriptSegment]`.
- **Outputs**: Merged `List[TranscriptSegment]`.
- **Critical Logic**: Merges segments < 4 words (either current OR next) if the gap is < 1.5s. NEVER merges across language boundaries. Caps merged duration at 30s for subtitle compatibility.

### `language_detector.py`
- **Role**: Classifies the language of a single segment.
- **Inputs**: Segment text, Whisper's initial language guess.
- **Outputs**: ISO-639-1 code (`ar`, `ur`, `en`, `und`).
- **Critical Logic**: Layered detection. Trusts Whisper for long segments, uses `langdetect` for fast passes, and relies on `lingua` and custom Urdu vocabulary/character regex sets to disambiguate Arabic from Urdu.

### `arabic_utils.py`
- **Role**: Centralized Arabic text normalization.
- **Inputs**: Raw string.
- **Outputs**: Normalized string.
- **Critical Logic**: Uses `str.maketrans` and Regex to strip all tashkeel (diacritics), normalize alef (أ إ آ ٱ → ا), normalize hamza (ؤ → و, ئ → ي), convert taa marbuta/alef maqsura, and strip bidirectional formatting characters. Crucial for matching Whisper output to classical texts. Includes an `lru_cache` wrapper for hot loops. Also houses the **canonical hallucination blocklist** used by both `transcriber.py` and `quran_matcher.py`.

### `quran_matcher.py`
- **Role**: Identifies Quranic ayahs and Islamic formulas.
- **Inputs**: Normalized Arabic string.
- **Outputs**: `QuranMatch`, `FormulaMatch`, or `None`.
- **Critical Logic**: O(1) exact match lookup dict for speed. Falls back to `rapidfuzz.fuzz.token_set_ratio` (threshold 0.88). Rejects fuzzy queries < 20 chars to prevent false positives.

### `hadith_matcher.py`
- **Role**: Matches text against Sunnah.com.
- **Inputs**: Normalized Arabic string.
- **Outputs**: `HadithMatch` or `None`.
- **Critical Logic**: Hits `GET /hadiths/search`. Uses exponential backoff for HTTP 429 Rate Limits. Uses `rapidfuzz.fuzz.partial_ratio` on the Arabic body. Implements context manager protocol (`with HadithMatcher() as m:`) for proper session cleanup.
- **Limitations**: API rate limits (100 req/hr) make it a massive bottleneck.

### `output_handler.py` & `flagged_handler.py`
- **Role**: Final data serialization.
- **Inputs**: `List[EnrichedSegment]`.
- **Outputs**: Disk I/O (TXT, JSON, SRT).
- **Critical Logic**: Formats text, adds `[UR]/[AR]/[EN]` language badges, injects `↳` annotations for verified matches, and routes failed Arabic matches to `flagged.txt` with word-level confidence scores for human reviewers.

---

## Transcription Pipeline

1. **Audio Load**: Processed via `faster-whisper`.
2. **VAD**: Silero VAD strips silence (`min_silence_duration_ms=300`, `speech_pad_ms=200`).
3. **Whisper Config**: `beam_size=5`, `repetition_penalty=1.15`, `compression_ratio_threshold=2.2`. `condition_on_previous_text=False` is critical to stop hallucinations from cascading.
4. **Multilingual Prompting**: An `initial_prompt` containing Urdu instructions, Arabic formulas, and English keywords is passed to prime the model to transcribe natively rather than translating.
5. **Post-processing**: Segments are deduplicated, hallucination phrases stripped, and short fragments merged.

---

## Arabic Processing Pipeline

1. **Trigger**: Segment is detected as `ar`.
2. **Normalization**: Segment passes through `normalise_arabic` to strip diacritics and standardize letters.
3. **Formula Check**: Checked against a hardcoded list of ~30 formulas (Salawat, Takbir).
4. **Quran Exact**: O(1) check against the local 6,236 ayah corpus.
5. **Quran Fuzzy**: Rapidfuzz `token_set_ratio` check against the corpus.
6. **Hadith Fallback**: If no Quran match, marked as `pending_hadith`.
7. **Async resolution**: `ThreadPoolExecutor` dispatches pending segments to the Sunnah.com API.
8. **Flagging**: If all matchers return `None`, the segment is sent to `flagged_handler`.

---

## Output System

- **`transcript.txt`**: The primary human-readable artifact. Features a rich header with duration and language stats. Segments include language badges and indented match annotations.
- **`transcript.json`**: Structured data dump of all `EnrichedSegment` objects for downstream processing or UI consumption.
- **`transcript.srt`**: Subtitle format export.
- **`flagged.txt`**: A specialized reviewer workbench. Contains unverified Arabic text, exact timestamps, and word-by-word Whisper confidence probabilities to help reviewers identify transcription errors.

---

## Current Whisper Settings

- **Model**: `large-v3` (Best for multilingual).
- **Compute**: `float16` on GPU, auto-downgrades to `int8` on CPU.
- **Beam Size**: `5`.
- **Repetition Penalty**: `1.15` (Aggressive to fight YouTube hallucination loops).
- **No Repeat N-Gram**: `4`.
- **VAD**: Enabled. `min_silence=300ms`, `speech_pad=200ms`.
- **Primary Language**: `ur` (Critical: forces transcription over translation).
- **Hardware Assumption**: NVIDIA GPU with >= 10GB VRAM (e.g., Colab T4).
- **GPU Detection**: Pipeline auto-detects GPU availability via `torch.cuda` and falls back to CPU+int8 gracefully.

---

## Environment Assumptions

- **Development**: Google Colab (Linux, NVIDIA T4 16GB).
- **Deployment**: Local Windows/Linux machine with CUDA.
- **Python**: 3.10+.
- **Dependencies**: Heavily relies on `faster-whisper` and CTranslate2. No raw PyTorch inference.

---

## Known Problems

- **API Bottleneck**: Sunnah.com API free tier (100 req/hr) is insufficient for long lectures.
- **Partial Ayah Detection**: Extremely short Quranic phrases (e.g. 2-3 words) are currently ignored by the fuzzy matcher (`MIN_FUZZY_QUERY_CHARS`) to prevent false positives, meaning some valid short quotes go unmatched.
- **Urdu/Arabic Ambiguity**: Whisper sometimes transcribes Urdu using standard Arabic characters, omitting Urdu-specific glyphs. This causes the language detector to flag the Urdu segment as Arabic, resulting in false positives in `flagged.txt`.
- **Dual Codebase State**: Root modules still exist alongside the `islamic_stt/` package. They need to be fully moved into `islamic_stt/core/`.
- **~~Credential Exposure~~**: *(Fixed)* API keys were previously passable via CLI args. Now env-only.

---

## Technical Debt

- **Missing SQLite Local Hadith Database**: The architecture desperately needs a local Hadith database with FTS5 search to replace the Sunnah.com API.
- **Missing Tests**: No comprehensive pytest suite for matchers and the pipeline.
- **Architecture Migration**: Phase 1 fixed the code, but files still need to be physically moved into the structured `islamic_stt/` package directory (Phase 2).
- **Hardcoded Prompts**: Initial prompt is hardcoded in `config.py` rather than being loaded dynamically based on the target audience.

---

## Future Improvements

1. **Local Hadith DB**: Ingest a dump of the top 6 major Hadith collections into SQLite.
2. **Semantic Search**: Implement Sentence-Transformers + FAISS to allow semantic matching of paraphrased Hadiths.
3. **Speaker Diarization**: Integrate `pyannote.audio` to distinguish between the lecturer and a dedicated Quran reciter.
4. **Desktop GUI**: Build a local PyQt6 or Electron app so non-technical users can run the STT pipeline.
5. **Markdown Export**: Generate beautiful Markdown files suitable for direct publication.

---

## Production Readiness Status

**Current Status: 7.5/10 (Phase 1 + Code Review Complete)**

*Strengths:*
- Inference is stable and GPU accelerated.
- Segment merging produces readable transcripts (now absorbs orphan words).
- Arabic normalization is highly robust.
- Comprehensive structured logging with rotating file output.
- Proper exception handling for CUDA OOM, corpus corruption, and network errors.
- GPU auto-detection with graceful CPU fallback.
- Input validation with file size and duration limits.
- Unified segment quality scoring.
- API credentials protected (env-only, no CLI exposure).
- All data models use proper type annotations (no `Optional[object]`).

*Weaknesses (Blocking 10/10):*
- Sunnah.com API reliance is a fatal flaw for scalability.
- Project lacks `pyproject.toml` / proper packaging.
- Root folder is cluttered.

---

## Coding Standards

- **Typing**: Strict `typing` annotations everywhere. Use `from __future__ import annotations`.
- **Dataclasses**: Heavily favored for state and data passing (`TranscriptSegment`, `EnrichedSegment`, `PipelineConfig`, `_FlaggedEntry`).
- **No `setattr` Hacks**: State tracking is done explicitly via defined class attributes (e.g., `pending_hadith: bool`).
- **Logging**: Use standard Python `logging` module. **Never use bare `print()`**. All modules use `logger = logging.getLogger(__name__)`.
- **Performance**: O(1) lookups via `set` and `dict` wherever possible. `lru_cache` used on hot paths (e.g., text normalization).
- **Single Source of Truth**: Shared constants (like the hallucination blocklist) live in `arabic_utils.py` and are imported, never duplicated.
- **Immutability**: Config objects passed to `run_pipeline()` are never mutated. Device overrides use local variables.

---

## Performance Considerations

- **VRAM**: `large-v3` + `float16` requires ~8-10GB VRAM.
- **CPU Bottlenecks**: Fuzzy matching 6,236 Quranic ayahs using Rapidfuzz takes time. We batch and pre-filter non-Arabic text to mitigate this.
- **Network I/O**: ThreadPoolExecutor is used to parallelize Sunnah.com API requests, but rate limits negate this advantage.
- **Segment Count**: The `segment_merger` successfully reduces matcher invocations by ~30-40% by combining fragmented Whisper outputs.

---

## Security Considerations

- **API Keys**: Sunnah.com key is loaded exclusively via environment variables (`SUNNAH_API_KEY`). **Never** pass via CLI arguments — they are visible in `ps aux` and `--help` output.
- **File Parsing**: `json.load` on the Quran corpus validates the schema (must be a list of 114 surahs). Integrity checks (SHA256) should be enforced.
- **Path Traversal**: Output paths are currently trusted. User input must be sanitized if this becomes a web service.
- **Resource Limits**: `max_file_size_mb` (500 MB) and `max_audio_duration_s` (4 hours) prevent OOM crashes from oversized inputs.

---

## Development Workflow

Future developers and LLMs should follow this workflow:
1. **Read this Context**: Always consult this file first to understand why systems are built the way they are.
2. **Respect `arabic_utils.py`**: It is the foundational text-processing module. If a match fails, debug the normalization output first.
3. **Use Dataclasses**: Do not pass loose dictionaries. Update the relevant Dataclass.
4. **Test on Colab**: Always run changes against a real audio snippet in a Colab T4 environment to verify VRAM usage and transcription quality.

---

## Recommended Next Steps

1. **Phase 2 (Architecture Consolidation)**: Move `transcriber.py`, `segment_merger.py`, `language_detector.py`, and `arabic_utils.py` into `islamic_stt/core/`. Move matchers into `islamic_stt/matchers/`. Move handlers into `islamic_stt/output/`.
2. **Phase 3 (Local Hadith)**: Build the SQLite FTS5 database to replace the Sunnah.com API.
3. **Phase 4 (Packaging)**: Add `pyproject.toml`, `.env` support, and a `pytest` suite.

---

## Important Engineering Notes For Future LLMs

> [!CAUTION]
> **CRITICAL WARNINGS FOR FUTURE LLMs**
>
> 1. **The Translation Bug**: Whisper's `language=None` parameter performs FILE-LEVEL language detection. If the audio starts with Quran, Whisper will detect `ar` and **translate** all subsequent Urdu/English into Arabic text. **You MUST pass `language="ur"` to force code-switching transcription.**
>
> 2. **Arabic/Urdu Ambiguity**: Both use the Arabic script. Whisper frequently drops Urdu-exclusive characters (like ے, ٹ). Do not rely solely on character sets for language detection; vocabulary sets and heuristics (like `lingua`) are required.
>
> 3. **Never modify `EnrichedSegment` dynamically**: Previous versions used `setattr(es, "_pending_hadith", True)`. This broke type checkers and IDEs. Use explicit boolean flags like `pending_hadith`.
>
> 4. **`condition_on_previous_text=False`**: Do not change this to `True`. In multilingual lectures, a hallucination in an Arabic segment will bleed into and corrupt the following Urdu segment if this is True.
>
> 5. **`arabic_utils.py` is Shared State**: Both `quran_matcher` and `hadith_matcher` rely on its strict normalization. Changing how a letter normalizes (e.g., Alef variants) will break exact matches in the Quran corpus if the corpus isn't re-normalized simultaneously. The hallucination blocklist also lives here — always update it in `arabic_utils.py`, never in individual modules.
>
> 6. **Never mutate `PipelineConfig`**: The pipeline function receives config by reference. Mutating it corrupts the caller's state. Use local variables for any resolved/overridden values.

