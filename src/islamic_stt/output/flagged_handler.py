"""
flagged_handler.py
------------------
Collects Arabic segments that could not be matched to the Quran corpus or
the Sunnah.com hadith database and writes them to a separate flagged.txt.

Purpose
-------
flagged.txt is a reviewer's workbench — a compact list of Arabic text
snippets that need human verification.  Each entry includes:

  - Timestamp (HH:MM:SS)
  - The raw transcribed text
  - Word-level tokens and their confidence scores (for spotting
    low-confidence transcription that may have caused the match failure)
  - A blank "Verified:" field so a reviewer can fill it in

Output format example
---------------------
────────────────────────────────────────────────────────────────────────
[00:04:17] وَمَا أَرْسَلْنَاكَ إِلَّا رَحْمَةً لِّلْعَالَمِينَ
  Words   : وما(0.94) أرسلناك(0.89) إلا(0.96) رحمة(0.91) للعالمين(0.88)
  Verified: ___________________________________________________________
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List

from islamic_stt.core.transcriber import TranscriptSegment, Transcriber

logger = logging.getLogger(__name__)


__all__ = ["FlaggedHandler"]


_DIVIDER = "─" * 72


@dataclass(frozen=True, slots=True)
class _FlaggedEntry:
    """Internal typed container for a single flagged segment."""
    segment: TranscriptSegment
    reason: str


class FlaggedHandler:
    """
    Accumulates flagged segments during pipeline processing and writes them
    to disk at the end.

    Parameters
    ----------
    flagged_path : Destination file path (e.g. 'flagged.txt').
    """

    def __init__(self, flagged_path: str) -> None:
        self.flagged_path = flagged_path
        self._entries: list[_FlaggedEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, segment: TranscriptSegment, reason: str = "No Quran/Hadith match") -> None:
        """
        Register *segment* as needing human review.

        Parameters
        ----------
        segment : The unmatched TranscriptSegment.
        reason  : Short description of why it was flagged.
        """
        self._entries.append(_FlaggedEntry(segment=segment, reason=reason))

    def write(self) -> None:
        """
        Write all accumulated flagged entries to flagged.txt.
        Does nothing if no entries were added (no file is created).
        """
        if not self._entries:
            logger.info("No flagged segments — flagged.txt not created.")
            return

        os.makedirs(os.path.dirname(self.flagged_path) or ".", exist_ok=True)

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("FLAGGED ARABIC SEGMENTS — REQUIRES HUMAN VERIFICATION")
        lines.append(
            "These segments were transcribed as Arabic but could not be\n"
            "matched to the Quran corpus or Sunnah.com hadith database.\n"
            "Fill in the 'Verified:' field after manual review."
        )
        lines.append("=" * 72)
        lines.append("")

        for i, entry in enumerate(self._entries, start=1):
            seg = entry.segment
            reason = entry.reason

            ts = Transcriber.format_timestamp(seg.start)
            ts_end = Transcriber.format_timestamp(seg.end)

            lines.append(_DIVIDER)
            lines.append(f"#{i}  [{ts} → {ts_end}]")
            lines.append(f"  Text   : {seg.text}")
            lines.append(f"  Reason : {reason}")

            # Word-level confidence breakdown (helps spot transcription errors)
            if seg.words:
                word_detail = "  Words  : " + "  ".join(
                    f"{w.word}({w.probability:.2f})" for w in seg.words
                )
                # Wrap long lines at ~70 chars
                lines.append(_wrap_word_detail(word_detail, width=72))

            lines.append(f"  Verified: {'_' * 55}")
            lines.append("")

        lines.append(_DIVIDER)
        lines.append(f"Total flagged: {len(self._entries)}")
        lines.append(_DIVIDER)

        with open(self.flagged_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info(
            "%d flagged segment(s) written → %s",
            len(self._entries), self.flagged_path,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of segments flagged so far."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_word_detail(line: str, width: int = 72, indent_width: int = 11) -> str:
    """
    Wrap a long word-detail line at *width* characters, indenting
    continuation lines by *indent_width* spaces (aligned under the first
    word token after "  Words  : ").
    """
    if len(line) <= width:
        return line

    indent = " " * indent_width
    parts = line.split("  ")
    wrapped_lines = []
    current = ""

    for part in parts:
        if not part:
            continue
        candidate = current + "  " + part if current else part
        if len(candidate) > width and current:
            wrapped_lines.append(current)
            current = indent + part
        else:
            current = candidate

    if current:
        wrapped_lines.append(current)

    return "\n".join(wrapped_lines)
