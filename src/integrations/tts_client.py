"""Azure AI Speech REST "Text to speech" API - the classic
cognitiveservices/v1 synthesize endpoint, verified live against a real
resource 2026-10-17 (see docs/SOURCES.md "Azure TTS voiceover"):

- POST https://<region>.tts.speech.microsoft.com/cognitiveservices/v1 with
  an SSML body and the subscription key passed directly as
  Ocp-Apim-Subscription-Key - no separate token exchange needed for this
  endpoint (confirmed: a real synthesize call with only that header
  returned 200 and a real, playable MP3).
- GET .../cognitiveservices/voices/list (also just the subscription key)
  returned the real voice catalogue for the resource's region; ar-TN-
  ReemNeural and ar-TN-HediNeural were both confirmed present with
  Status "GA" for the real eastus resource used here - voice hosting is
  NOT limited to a handful of regions as one might guess from the general
  region/feature table, it was checked directly instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from xml.sax.saxutils import escape

DEFAULT_OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"


class TTSError(Exception):
    pass


# Owner's report, 2026-10-24: the voices read "like a book", with no
# grief. Checked on Microsoft's own voice table (learn.microsoft.com,
# speech-service/language-support, read 2026-10-24): every ar-* voice,
# ar-TN-ReemNeural and ar-TN-HediNeural included, has an EMPTY
# Style/Roles column - <mstts:express-as style="sad"> or crying is not
# available for them. Standard SSML <prosody> (rate, pitch) works for
# every neural voice, so a grieving line is read slower and lower.
@dataclass(frozen=True)
class Prosody:
    rate: str
    pitch: str


GRIEF_PROSODY = Prosody(rate="-15%", pitch="-8%")


class TTSClient(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str, *, lang: str, prosody: Prosody | None = None) -> bytes:
        """Returns real MP3 bytes for `text` read by `voice` (e.g.
        ar-TN-HediNeural), wrapped in <prosody> when given. Raises
        TTSError on a non-200 response."""


class RealAzureTTSClient(TTSClient):
    def __init__(self, subscription_key: str, region: str):
        self._subscription_key = subscription_key
        self._base_url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def synthesize(self, text: str, voice: str, *, lang: str, prosody: Prosody | None = None) -> bytes:
        import httpx

        spoken = escape(text)
        if prosody is not None:
            spoken = f"<prosody rate='{prosody.rate}' pitch='{prosody.pitch}'>{spoken}</prosody>"
        ssml = (
            f"<speak version='1.0' xml:lang='{lang}'>"
            f"<voice name='{voice}'>{spoken}</voice>"
            f"</speak>"
        )
        response = httpx.post(
            self._base_url,
            content=ssml.encode("utf-8"),
            headers={
                "Ocp-Apim-Subscription-Key": self._subscription_key,
                "Content-Type": "application/ssml+xml; charset=utf-8",
                "X-Microsoft-OutputFormat": DEFAULT_OUTPUT_FORMAT,
                "User-Agent": "nexus-studio",
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise TTSError(f"Azure TTS synthesize failed ({response.status_code}): {response.text[:500]}")
        return response.content


class MockTTSClient(TTSClient):
    """Returns fixed fake MP3 bytes and records every call - lets a test
    drive a whole episode's pipeline without a real Azure resource."""

    def __init__(self, scripted_bytes: bytes = b"fake-mp3-bytes"):
        self._scripted_bytes = scripted_bytes
        self.calls: list[tuple[str, str, str]] = []
        self.prosodies: list[Prosody | None] = []

    def synthesize(self, text: str, voice: str, *, lang: str, prosody: Prosody | None = None) -> bytes:
        self.calls.append((text, voice, lang))
        self.prosodies.append(prosody)
        return self._scripted_bytes
