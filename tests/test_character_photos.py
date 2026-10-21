from __future__ import annotations

from src.workflows.character_photos import resolve_placeholder_voice


def test_resolve_placeholder_voice_uses_the_real_tunisian_voices():
    # Owner's instruction, 2026-10-21: reversed from Syrian back to
    # Tunisian after a real produced episode's own audio proved the
    # Syrian dialect genuinely unclear to the owner - see
    # character_photos.py's own docstring for the full real history.
    # Both are real, live-verified Tunisian Arabic Azure voices
    # (docs/SOURCES.md "Azure TTS voiceover"), not a guess.
    assert resolve_placeholder_voice("male") == "ar-TN-HediNeural"
    assert resolve_placeholder_voice("female") == "ar-TN-ReemNeural"


def test_resolve_placeholder_voice_is_case_and_whitespace_insensitive():
    assert resolve_placeholder_voice(" Male ") == "ar-TN-HediNeural"
    assert resolve_placeholder_voice("FEMALE") == "ar-TN-ReemNeural"


def test_resolve_placeholder_voice_falls_back_to_male_for_an_unknown_gender():
    assert resolve_placeholder_voice("unknown") == "ar-TN-HediNeural"
