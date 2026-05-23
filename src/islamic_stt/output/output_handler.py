"""
output_handler.py
-----------------
Assembles all processed segments into a clean plain-text transcript and
writes it to disk.

Transcript format
-----------------
Each segment is written as:

    [HH:MM:SS] <text>

For Arabic segments where a Quran or Hadith match was found, an inline
annotation is appended:

    [HH:MM:SS] بسم الله الرحمن الرحيم
               ↳ [Quran 1:1 — Al-Fatihah, confidence: 100%]

    [HH:MM:SS] إنما الأعمال بالنيات
               ↳ [Hadith — Bukhari #1, confidence: 89%]

If no match was found the segment is still included in the transcript but
carries a ⚠ marker so the reviewer knows it was sent to flagged.txt.

    [HH:MM:SS] <unverified Arabic text>  ⚠ [see flagged.txt]

Changes (audit fixes)
---------------------
- Fix 7  : `@dataclass(slots=True)` on EnrichedSegment for faster attribute access.
- Fix 8  : `orjson` used for JSON serialization (3–10x faster than stdlib json);
           graceful fallback to stdlib json if not installed.
- Fix 10 : Modernised type annotations (list[X], X | None).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

from islamic_stt.core.transcriber import Transcriber
from islamic_stt.core.types import TranscriptSegment
from islamic_stt.matchers.quran_matcher import FormulaMatch, QuranMatch

# HadithMatch may come from either the API matcher or local DB matcher
try:
    from islamic_stt.matchers.hadith_matcher import HadithMatch
except ImportError:
    HadithMatch = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

# Fix 8: orjson with stdlib fallback
try:
    import orjson as _json_lib

    _ORJSON = True
except ImportError:
    import json as _json_lib  # type: ignore[no-redef]

    _ORJSON = False


__all__ = ["OutputHandler", "EnrichedSegment"]


# ---------------------------------------------------------------------------
# Data model for a fully enriched segment (Fix 7 — slots=True)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EnrichedSegment:
    """
    Carries everything needed to render one output line.

    Attributes
    ----------
    segment         : The raw TranscriptSegment from the transcriber.
    detected_lang   : ISO-639-1 code after language_detector processing.
    quran_match     : QuranMatch or None.
    hadith_match    : HadithMatch or None.
    formula_match   : FormulaMatch or None.
    is_flagged      : True if Arabic and neither Quran nor Hadith matched.
    pending_hadith  : True if awaiting async Hadith matching.
    """

    segment: TranscriptSegment
    detected_lang: str
    quran_match: QuranMatch | None = None
    hadith_match: HadithMatch | None = None
    formula_match: FormulaMatch | None = None
    is_flagged: bool = False
    pending_hadith: bool = False
    # P1 review 3: provenance tracking
    raw_text: str = ""  # original ASR output before post-processing
    was_corrected: bool = False  # True if post-processing changed the text
    # Hadith verification metadata (Stage 1: suggestion only)
    hadith_retrieval_confidence: float = 0.0  # calibrated retrieval score
    hadith_correction_type: str = ""  # "suggestion" | "none" | "rejected"
    hadith_correction_detail: str = ""  # human-readable suggestion/reason
    hadith_suggestion_safe: bool = False  # whether all safety gates passed

    @property
    def timestamp(self) -> str:
        return Transcriber.format_timestamp(self.segment.start)


# ---------------------------------------------------------------------------
# OutputHandler
# ---------------------------------------------------------------------------


class OutputHandler:
    """
    Writes the final transcript to a plain .txt file.

    Parameters
    ----------
    output_path : Destination file path (e.g. 'transcript.txt').
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def write(self, enriched_segments: list[EnrichedSegment]) -> None:
        """
        Serialise all enriched segments into a plain-text transcript.

        The file is UTF-8 encoded so that Arabic / Urdu text renders
        correctly in any modern editor.
        """
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        # --- Compute summary stats for the header -------------------------
        lang_counts: dict[str, int] = {}
        for es in enriched_segments:
            lang_counts[es.detected_lang] = lang_counts.get(es.detected_lang, 0) + 1

        duration_s = enriched_segments[-1].segment.end if enriched_segments else 0.0
        duration_str = Transcriber.format_timestamp(duration_s)

        lines: list[str] = []
        lines.append("═" * 72)
        lines.append(" ISLAMIC LECTURE TRANSCRIPT")
        lines.append(f" Duration: {duration_str}  |  Segments: {len(enriched_segments)}")
        lang_summary = "  ".join(f"{k.upper()}: {v}" for k, v in sorted(lang_counts.items()))
        lines.append(f" Languages: {lang_summary}")
        lines.append("═" * 72)
        lines.append("")

        # --- Language badge helper ----------------------------------------
        _LANG_BADGE = {"ar": "[AR]", "ur": "[UR]", "en": "[EN]", "und": "[??]"}

        for es in enriched_segments:
            ts = es.timestamp
            text = es.segment.text
            badge = _LANG_BADGE.get(es.detected_lang, f"[{es.detected_lang.upper()}]")

            # --- Build annotation line (only for Arabic segments) ----------
            annotation = ""

            if es.formula_match:
                m = es.formula_match
                annotation = f"           ↳ ☪ {m.label}"

            elif es.quran_match:
                m = es.quran_match
                pct = int(m.confidence * 100)
                exact_flag = " ✓exact" if m.is_exact else ""
                corrected_flag = ""
                if es.was_corrected:
                    corrected_flag = " [corrected]"
                # Show matched span if it differs from full segment text
                span_flag = ""
                if m.matched_text and m.matched_text != text:
                    span_preview = m.matched_text[:60] + ("…" if len(m.matched_text) > 60 else "")
                    span_flag = f' [span: "{span_preview}"]'

                alt_refs = getattr(m, "alternate_refs", None)

                if getattr(m, "is_ambiguous", False) and alt_refs:
                    # P0 review 5: Do NOT render a single authoritative citation.
                    # Show all candidates equally so no single ref is misread.
                    alt_strs = [f"{r['surah_id']}:{r['ayah_id']}" for r in alt_refs[:8]]
                    if len(alt_refs) > 8:
                        alt_strs.append(f"…+{len(alt_refs) - 8} more")
                    annotation = (
                        f"           ↳ 📖 Quran [ambiguous — {m.ambiguous_count} ayahs] "
                        f"({pct}%) possible: {', '.join(alt_strs)}"
                        f"{corrected_flag}{span_flag}"
                    )
                else:
                    annotation = (
                        f"           ↳ 📖 Quran {m.surah_id}:{m.ayah_id} — "
                        f"{m.surah_name}{exact_flag} ({pct}%)"
                        f"{corrected_flag}{span_flag}"
                    )

            elif es.hadith_match:
                m = es.hadith_match
                pct = int(m.confidence * 100)
                collection = m.collection.capitalize()
                paraphrase_flag = ""
                if hasattr(m, "is_paraphrase") and m.is_paraphrase:
                    paraphrase_flag = " ⚠ paraphrase"
                annotation = (
                    f"           ↳ 📜 Hadith — {collection} #{m.hadith_number} "
                    f"({pct}%){paraphrase_flag}"
                )

            elif es.is_flagged:
                # Arabic but unverified — add warning marker
                text = text + "  ⚠ [unverified — see flagged.txt]"

            # --- Emit segment line ----------------------------------------
            lines.append(f"[{ts}] {badge} {text}")
            if annotation:
                lines.append(annotation)

        lines.append("")
        lines.append("═" * 72)
        lines.append(f" Total segments: {len(enriched_segments)}")
        lines.append("═" * 72)

        with open(self.output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info("Transcript written → %s", self.output_path)
        self._write_json(enriched_segments)
        self._write_srt(enriched_segments)

    def _write_json(self, enriched_segments: list[EnrichedSegment]) -> None:
        """Write JSON transcript using orjson when available (Fix 8)."""
        json_path = os.path.splitext(self.output_path)[0] + ".json"
        payload = []
        for es in enriched_segments:
            hadith_data = None
            if es.hadith_match:
                try:
                    hadith_data = asdict(es.hadith_match)
                except Exception:
                    hadith_data = {"matched_text": str(es.hadith_match)}
            quran_data = None
            if es.quran_match:
                quran_data = asdict(es.quran_match)
            # Hadith verification metadata (Stage 1: suggestion only)
            hadith_verification_data = None
            if es.hadith_correction_type:
                hadith_verification_data = {
                    "retrieval_confidence": es.hadith_retrieval_confidence,
                    "correction_type": es.hadith_correction_type,
                    "correction_detail": es.hadith_correction_detail,
                    "suggestion_safe": es.hadith_suggestion_safe,
                    "applied": False,  # Stage 1: never applied
                }
            payload.append(
                {
                    "start": es.segment.start,
                    "end": es.segment.end,
                    "timestamp": es.timestamp,
                    "text": es.segment.text,
                    "raw_text": es.raw_text if es.was_corrected else None,
                    "was_corrected": es.was_corrected,
                    "detected_language": es.detected_lang,
                    "avg_word_confidence": es.segment.avg_word_confidence,
                    "is_flagged": es.is_flagged,
                    "quran_match": quran_data,
                    "hadith_match": hadith_data,
                    "hadith_verification": hadith_verification_data,
                    "formula_match": asdict(es.formula_match) if es.formula_match else None,
                }
            )
        if _ORJSON:
            with open(json_path, "wb") as fh:
                fh.write(_json_lib.dumps(payload, option=_json_lib.OPT_INDENT_2))
        else:
            with open(json_path, "w", encoding="utf-8") as fh:
                _json_lib.dump(payload, fh, ensure_ascii=False, indent=2)  # type: ignore[attr-defined]
        logger.info("JSON transcript written → %s", json_path)

    def _write_srt(self, enriched_segments: list[EnrichedSegment]) -> None:
        srt_path = os.path.splitext(self.output_path)[0] + ".srt"
        lines: list[str] = []
        for idx, es in enumerate(enriched_segments, start=1):
            lines.append(str(idx))
            start = Transcriber.format_timestamp(es.segment.start, milliseconds=True).replace(
                ".", ","
            )
            end = Transcriber.format_timestamp(es.segment.end, milliseconds=True).replace(".", ",")
            lines.append(f"{start} --> {end}")
            lang = es.detected_lang.upper()
            text = f"[{lang}] {es.segment.text}"
            if es.quran_match:
                text += f"\n[QURAN {es.quran_match.surah_id}:{es.quran_match.ayah_id}]"
            elif es.hadith_match:
                text += f"\n[HADITH {es.hadith_match.collection} #{es.hadith_match.hadith_number}]"
            elif es.formula_match:
                text += f"\n[FORMULA {es.formula_match.label}]"
            lines.append(text)
            lines.append("")

        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("SRT transcript written → %s", srt_path)
