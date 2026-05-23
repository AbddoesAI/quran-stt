"""
Islamic STT — AI-powered Islamic lecture transcription.

Provides Quran & Hadith detection for multilingual (Urdu/Arabic/English) audio.
"""

__version__ = "0.2.0"


def run_pipeline(*args, **kwargs):
    """Lazy wrapper — imports the real pipeline only when called."""
    from islamic_stt.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


__all__ = ["run_pipeline", "__version__"]
