from __future__ import annotations

import pytest

from src.integrations.fal_video_client import (
    ENDPOINT_ID,
    MockFalVideoClient,
    FalVideoError,
    RealFalVideoClient,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict | None = None, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


def test_generate_speech_to_video_submits_polls_and_downloads_the_real_clip(monkeypatch):
    calls = {"get": []}

    def fake_post(url, *, headers, json, timeout):
        assert url == f"https://queue.fal.run/{ENDPOINT_ID}"
        assert headers["Authorization"] == "Key real-fal-key"
        assert json == {
            "prompt": "smiling, looking at camera",
            "image_url": "data:image/jpeg;base64,aW1n",
            "audio_url": "data:audio/mpeg;base64,YXVk",
            "resolution": "480p",
        }
        return _FakeResponse(
            status_code=200,
            json_data={
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
                "cancel_url": "https://queue.fal.run/x/requests/req-1/cancel",
            },
        )

    def fake_get(url, *, headers=None, timeout=30):
        calls["get"].append(url)
        if url.endswith("/status"):
            return _FakeResponse(status_code=200, json_data={"status": "COMPLETED"})
        if url == "https://queue.fal.run/x/requests/req-1":
            return _FakeResponse(status_code=200, json_data={"video": {"url": "https://cdn.fal.media/x.mp4"}})
        if url == "https://cdn.fal.media/x.mp4":
            return _FakeResponse(status_code=200, content=b"real-mp4-bytes")
        raise AssertionError(f"unexpected GET {url}")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealFalVideoClient("real-fal-key")
    data = client.generate_speech_to_video(
        prompt="smiling, looking at camera",
        image_data_uri="data:image/jpeg;base64,aW1n",
        audio_data_uri="data:audio/mpeg;base64,YXVk",
    )

    assert data == b"real-mp4-bytes"


def test_generate_speech_to_video_polls_through_in_progress_before_completed(monkeypatch):
    statuses = iter(["IN_QUEUE", "IN_PROGRESS", "COMPLETED"])

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(
            status_code=200,
            json_data={
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )

    def fake_get(url, *, headers=None, timeout=30):
        if url.endswith("/status"):
            return _FakeResponse(status_code=200, json_data={"status": next(statuses)})
        if url == "https://queue.fal.run/x/requests/req-1":
            return _FakeResponse(status_code=200, json_data={"video": {"url": "https://cdn.fal.media/x.mp4"}})
        return _FakeResponse(status_code=200, content=b"bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealFalVideoClient("key")
    data = client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")

    assert data == b"bytes"


def test_generate_speech_to_video_raises_on_a_failed_job(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(
            status_code=200,
            json_data={
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )

    def fake_get(url, *, headers=None, timeout=30):
        return _FakeResponse(status_code=200, json_data={"status": "ERROR"}, text='{"status": "ERROR"}')

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealFalVideoClient("key")
    with pytest.raises(FalVideoError, match="failed"):
        client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")


def test_generate_speech_to_video_raises_on_a_non_2xx_submit(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=401, text="invalid key")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealFalVideoClient("bad-key")
    with pytest.raises(FalVideoError, match="401"):
        client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")


def test_mock_fal_video_client_records_calls_and_returns_scripted_bytes():
    mock = MockFalVideoClient(scripted_bytes=b"scripted-clip")

    data = mock.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a", resolution="720p")

    assert data == b"scripted-clip"
    assert mock.calls == [{"prompt": "p", "image_data_uri": "i", "audio_data_uri": "a", "resolution": "720p"}]
