from __future__ import annotations

import pytest

from src.integrations.tts_client import DEFAULT_OUTPUT_FORMAT, MockTTSClient, RealAzureTTSClient, TTSError


class _FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text


def test_synthesize_posts_the_real_url_headers_and_ssml(monkeypatch):
    captured = {}

    def fake_post(url, *, content, headers, timeout):
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = headers
        return _FakeResponse(status_code=200, content=b"\xff\xfbreal-mp3-bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealAzureTTSClient("real-key-123", "eastus")
    data = client.synthesize("مرحبا", "ar-TN-HediNeural", lang="ar-TN")

    assert data == b"\xff\xfbreal-mp3-bytes"
    assert captured["url"] == "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"
    assert captured["headers"]["Ocp-Apim-Subscription-Key"] == "real-key-123"
    assert captured["headers"]["Content-Type"] == "application/ssml+xml; charset=utf-8"
    assert captured["headers"]["X-Microsoft-OutputFormat"] == DEFAULT_OUTPUT_FORMAT
    ssml = captured["content"].decode("utf-8")
    assert "xml:lang='ar-TN'" in ssml
    assert "voice name='ar-TN-HediNeural'" in ssml
    assert "مرحبا" in ssml


def test_synthesize_escapes_xml_special_characters_in_the_text(monkeypatch):
    captured = {}

    def fake_post(url, *, content, headers, timeout):
        captured["content"] = content
        return _FakeResponse(status_code=200, content=b"bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealAzureTTSClient("key", "eastus")
    client.synthesize("Tom & Jerry <say \"hi\">", "ar-TN-HediNeural", lang="ar-TN")

    ssml = captured["content"].decode("utf-8")
    assert "&amp;" in ssml
    assert "<say" not in ssml  # the literal text's '<' must be escaped, not left as a tag


def test_synthesize_raises_tts_error_on_a_non_200_response(monkeypatch):
    def fake_post(url, *, content, headers, timeout):
        return _FakeResponse(status_code=401, text="Access denied due to invalid subscription key")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealAzureTTSClient("bad-key", "eastus")
    with pytest.raises(TTSError, match="401"):
        client.synthesize("hello", "ar-TN-HediNeural", lang="ar-TN")


def test_mock_tts_client_records_calls_and_returns_scripted_bytes():
    mock = MockTTSClient(scripted_bytes=b"scripted-audio")

    data = mock.synthesize("hello", "ar-TN-ReemNeural", lang="ar-TN")

    assert data == b"scripted-audio"
    assert mock.calls == [("hello", "ar-TN-ReemNeural", "ar-TN")]


def test_synthesize_wraps_the_text_in_prosody_when_given(monkeypatch):
    from src.integrations.tts_client import GRIEF_PROSODY

    captured = {}

    def fake_post(url, *, content, headers, timeout):
        captured["content"] = content
        return _FakeResponse(status_code=200, content=b"mp3")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    RealAzureTTSClient("k", "eastus").synthesize("سامحيني", "ar-TN-HediNeural", lang="ar-TN", prosody=GRIEF_PROSODY)

    assert "<voice name='ar-TN-HediNeural'><prosody rate='-15%' pitch='-8%'>سامحيني</prosody></voice>" in captured["content"].decode("utf-8")
