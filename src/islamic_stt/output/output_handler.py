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
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from islamic_stt.core.transcriber import TranscriptSegment, Transcriber
from islamic_stt.matchers.quran_matcher import FormulaMatch, QuranMatch
from islamic_stt.matchers.hadith_matcher import HadithMatch

logger = logging.getLogger(__name__)


__all__ = ["OutputHandler", "EnrichedSegment"]


# ---------------------------------------------------------------------------
# Data model for a fully enriched segment
# ---------------------------------------------------------------------------

@dataclass
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
    quran_match: Optional[QuranMatch] = None
    hadith_match: Optional[HadithMatch] = None
    formula_match: Optional[FormulaMatch] = None
    is_flagged: bool = False
    pending_hadith: bool = False

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

    def write(self, enriched_segments: List[EnrichedSegment]) -> None:
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

        duration_s = (
            enriched_segments[-1].segment.end if enriched_segments else 0.0
        )
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
                annotation = (
                    f"           ↳ 📖 Quran {m.surah_id}:{m.ayah_id} — "
                    f"{m.surah_name}{exact_flag} ({pct}%)"
                )

            elif es.hadith_match:
                m = es.hadith_match
                pct = int(m.confidence * 100)
                collection = m.collection.capitalize()
                annotation = (
                    f"           ↳ 📜 Hadith — {collection} #{m.hadith_number} "
                    f"({pct}%)"
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

    def _write_json(self, enriched_segments: List[EnrichedSegment]) -> None:
        json_path = os.path.splitext(self.output_path)[0] + ".json"
        payload = []
        for es in enriched_segments:
            payload.append(
                {
                    "start": es.segment.start,
                    "end": es.segment.end,
                    "timestamp": es.timestamp,
                    "text": es.segment.text,
                    "detected_language": es.detected_lang,
                    "avg_word_confidence": es.segment.avg_word_confidence,
                    "is_flagged": es.is_flagged,
                    "quran_match": asdict(es.quran_match) if es.quran_match else None,
                    "hadith_match": asdict(es.hadith_match) if es.hadith_match else None,
                    "formula_match": asdict(es.formula_match) if es.formula_match else None,
                }
            )
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("JSON transcript written → %s", json_path)

    def _write_srt(self, enriched_segments: List[EnrichedSegment]) -> None:
        srt_path = os.path.splitext(self.output_path)[0] + ".srt"
        lines: list[str] = []
        for idx, es in enumerate(enriched_segments, start=1):
            lines.append(str(idx))
            start = Transcriber.format_timestamp(es.segment.start, milliseconds=True).replace(".", ",")
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
