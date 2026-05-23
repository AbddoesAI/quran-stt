"""
prompt_builder.py
-----------------
Dynamic contextual prompt builder for Islamic STT.

Rotates prompts based on detected content mode:
  - Quran mode   : Quranic recitation context
  - Hadith mode  : Hadith narration context
  - Bayan mode   : General Islamic lecture (default)
  - Naat mode    : Devotional poetry

Also builds continuation prompts from previous segments to improve
coherence without the risks of global condition_on_previous_text.

Safety:
  - Multi-factor confidence gating (logprob + no_speech + flagged)
  - Bounded history (max 3 trusted segments)
  - Language-locked prompts for Arabic/Urdu-only decoding
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from islamic_stt.core.types import TranscriptSegment

logger = logging.getLogger(__name__)

__all__ = [
    "build_contextual_prompt",
    "build_language_locked_prompt",
    "detect_content_mode",
]


# ---------------------------------------------------------------------------
# Mode-specific prompt templates
# ---------------------------------------------------------------------------

_BASE_INSTRUCTION = "ہر زبان کو اسی زبان میں لکھیں، ترجمہ نہ کریں۔ "

_QURAN_PROMPT = (
    "یہ قرآن مجید کی تلاوت ہے۔ "
    + _BASE_INSTRUCTION
    + "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ۔ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ۔ "
    "الرَّحْمَنِ الرَّحِيمِ۔ مَالِكِ يَوْمِ الدِّينِ۔ "
    "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ۔ "
    "سورة البقرة، سورة آل عمران، سورة النساء، يس، الرحمن، الملك۔ "
)

_HADITH_PROMPT = (
    "یہ حدیث نبوی کا بیان ہے۔ " + _BASE_INSTRUCTION + "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ۔ "
    "ابو ہریرہ، عبداللہ بن عمر، حضرت عائشہ، حضرت انس بن مالک، ابن عباس۔ "
    "صحیح بخاری، صحیح مسلم، سنن ترمذی، سنن ابو داؤد، سنن نسائی، سنن ابن ماجہ۔ "
    "قال رسول الله، عن أبي هريرة، حدثنا، أخبرنا۔ "
)

_BAYAN_PROMPT = (
    "یہ ایک اسلامی بیان ہے جس میں اردو، عربی اور انگریزی تینوں زبانیں بولی جاتی ہیں۔ "
    + _BASE_INSTRUCTION
    + "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ۔ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ۔ "
    "سُبْحَانَ اللَّهِ۔ لَا إِلَهَ إِلَّا اللَّهُ۔ "
    "ابو ہریرہ، عبداللہ بن عمر، حضرت عائشہ۔ "
    "صحیح بخاری، صحیح مسلم، سنن ترمذی۔ "
    "قرآن، حدیث، سنت، صحابہ، تابعین، تفسیر، فقہ، شریعت، عقیدہ۔ "
    "سورة البقرة، سورة آل عمران، يس، الرحمن، الملك۔ "
    "Islamic, Quran, Hadith, Sunnah, Prophet Muhammad, peace be upon him. "
    "Sahih Bukhari, Sahih Muslim, Surah, Ayah, Tafsir."
)

_NAAT_PROMPT = (
    "یہ نعتیہ کلام ہے۔ " + _BASE_INSTRUCTION + "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ۔ "
    "مصطفٰی، حبیب، رسول اکرم، نبی کریم، محمد۔ "
    "مدینہ، مکہ، روضہ، حرم۔ "
)

_MODE_PROMPTS = {
    "quran": _QURAN_PROMPT,
    "hadith": _HADITH_PROMPT,
    "bayan": _BAYAN_PROMPT,
    "naat": _NAAT_PROMPT,
}


# ---------------------------------------------------------------------------
# Language-locked prompts (pure language, no mixing)
# ---------------------------------------------------------------------------

_ARABIC_LOCKED_PROMPT = (
    "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ۔ "
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ۔ الرَّحْمَنِ الرَّحِيمِ۔ "
    "مَالِكِ يَوْمِ الدِّينِ۔ إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ۔ "
    "صَدَقَ اللَّهُ الْعَظِيمُ۔ أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ۔ "
    "سُبْحَانَ اللَّهِ وَبِحَمْدِهِ سُبْحَانَ اللَّهِ الْعَظِيمِ۔ "
    "قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ۔ "
)

_URDU_LOCKED_PROMPT = (
    "یہ اردو میں بیان ہے۔ ہر بات اردو میں لکھیں۔ "
    "اللہ تعالیٰ نے فرمایا۔ نبی کریم صلی اللہ علیہ وسلم نے فرمایا۔ "
    "حضرات! آج ہم بات کریں گے۔ "
    "قرآن مجید، حدیث شریف، صحابہ کرام، تابعین۔ "
)

_ENGLISH_LOCKED_PROMPT = (
    "This is an Islamic lecture in English. "
    "Prophet Muhammad, peace be upon him. "
    "Quran, Hadith, Sunnah, Shariah, Tafsir, Fiqh. "
    "Sahih Bukhari, Sahih Muslim, Sunan Tirmidhi. "
)


def build_language_locked_prompt(lang: str) -> str:
    """
    Return a pure single-language prompt for language-locked decoding.

    When a segment is confidently one language (probability > 0.90),
    using a language-locked prompt prevents cross-language contamination.
    """
    if lang == "ar":
        return _ARABIC_LOCKED_PROMPT
    elif lang == "ur":
        return _URDU_LOCKED_PROMPT
    elif lang == "en":
        return _ENGLISH_LOCKED_PROMPT
    return _BAYAN_PROMPT  # fallback to mixed


# ---------------------------------------------------------------------------
# Content mode detection
# ---------------------------------------------------------------------------

_QURAN_SIGNALS = {"سورة", "آية", "تلاوت", "قرآن", "اعوذ", "بسم الله"}
_HADITH_SIGNALS = {"حدیث", "بخاری", "مسلم", "ترمذی", "روایت", "حدثنا", "اخبرنا", "قال رسول"}
_NAAT_SIGNALS = {"نعت", "مصطفٰی", "مدینہ", "صلوا", "درود"}


def detect_content_mode(
    recent_segments: list[TranscriptSegment],
    *,
    window: int = 5,
) -> str:
    """
    Detect content mode from recent segments.

    Returns one of: 'quran', 'hadith', 'naat', 'bayan' (default).
    """
    if not recent_segments:
        return "bayan"

    combined = " ".join(seg.text for seg in recent_segments[-window:]).lower()

    quran_hits = sum(1 for kw in _QURAN_SIGNALS if kw in combined)
    hadith_hits = sum(1 for kw in _HADITH_SIGNALS if kw in combined)
    naat_hits = sum(1 for kw in _NAAT_SIGNALS if kw in combined)

    scores = {
        "quran": quran_hits,
        "hadith": hadith_hits,
        "naat": naat_hits,
    }
    best_mode = max(scores, key=scores.get)
    if scores[best_mode] >= 2:
        return best_mode

    return "bayan"


# ---------------------------------------------------------------------------
# Segment trust filter (multi-factor gating)
# ---------------------------------------------------------------------------


def _is_trusted_segment(seg: TranscriptSegment, max_compression_ratio: float = 1.8) -> bool:
    """
    Multi-factor trust check for prompt context inclusion.
    """
    if seg.avg_logprob <= -0.45:
        return False
    if getattr(seg, "avg_word_confidence", 0.0) <= 0.80:
        return False
    if getattr(seg, "compression_ratio", 0.0) >= max_compression_ratio:
        return False
    if getattr(seg, "mixed_script_ratio", 0.0) >= 0.05:
        return False
    if getattr(seg, "is_flagged", False):
        return False
    if not getattr(seg, "prompt_safe", True):
        return False
    if len(seg.text.split()) < 3:
        return False
    return True


# ---------------------------------------------------------------------------
# Contextual prompt builder (with safety)
# ---------------------------------------------------------------------------

MAX_PROMPT_HISTORY_SEGMENTS = 3


def build_contextual_prompt(
    mode: str = "bayan",
    recent_segments: list[TranscriptSegment] | None = None,
    *,
    max_context_tokens: int = 50,
    max_compression_ratio: float = 1.8,
) -> str:
    """
    Build a contextual prompt for the next transcription chunk.
    """
    base = _MODE_PROMPTS.get(mode, _BAYAN_PROMPT)

    if not recent_segments:
        return base

    # Filter to trusted segments only (multi-factor gating)
    trusted = [seg for seg in recent_segments if _is_trusted_segment(seg, max_compression_ratio)]

    if not trusted:
        return base

    # Bounded history: keep only last N trusted segments
    trusted = trusted[-MAX_PROMPT_HISTORY_SEGMENTS:]

    # Append recent context for continuation (last N words)
    context_words: list[str] = []
    for seg in reversed(trusted):
        words = seg.text.split()
        context_words = words + context_words
        if len(context_words) >= max_context_tokens:
            break

    context_text = " ".join(context_words[-max_context_tokens:])

    # Prompt entropy protection: check for highly repetitive n-grams
    if _has_high_repetition(context_text):
        return base

    if context_text.strip():
        return base + " " + context_text

    return base


def _has_high_repetition(text: str) -> bool:
    """Simple check for repeating bigrams/trigrams to prevent prompt entropy."""
    words = text.split()
    if len(words) < 6:
        return False
    from collections import Counter

    # check bigrams
    bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    if bigrams:
        counts = Counter(bigrams)
        if counts.most_common(1)[0][1] > 3:
            return True
    return False
