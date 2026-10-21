from __future__ import annotations

import pytest

from src.integrations.replicate_video_client import (
    ENDPOINT,
    MockReplicateVideoClient,
    ReplicateVideoError,
    RealReplicateVideoClient,
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
    def fake_post(url, *, headers, json, timeout):
        assert url == ENDPOINT
        assert headers["Authorization"] == "Bearer real-replicate-token"
        assert json == {"input": {"prompt": "smiling, looking at camera", "image": "data:image/jpeg;base64,aW1n", "audio": "data:audio/mpeg;base64,YXVk"}}
        return _FakeResponse(
            status_code=201,
            json_data={"id": "pred-1", "status": "starting", "urls": {"get": "https://api.replicate.com/v1/predictions/pred-1"}},
        )

    def fake_get(url, *, headers=None, timeout=30):
        assert url == "https://api.replicate.com/v1/predictions/pred-1" or url == "https://real.cdn/clip.mp4"
        if url == "https://api.replicate.com/v1/predictions/pred-1":
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": "https://real.cdn/clip.mp4"})
        return _FakeResponse(status_code=200, content=b"real-mp4-bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealReplicateVideoClient("real-replicate-token")
    data = client.generate_speech_to_video(
        prompt="smiling, looking at camera",
        image_data_uri="data:image/jpeg;base64,aW1n",
        audio_data_uri="data:audio/mpeg;base64,YXVk",
    )

    assert data == b"real-mp4-bytes"


def test_generate_speech_to_video_ignores_the_resolution_kwarg(monkeypatch):
    # Real finding, 2026-10-21: unlike fal.ai, this model's own real
    # Playground schema (owner's screenshot) has no resolution field at
    # all. resolution is accepted only so this client matches
    # FalVideoClient's interface - a real request must never send it.
    def fake_post(url, *, headers, json, timeout):
        assert "resolution" not in json["input"]
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        if url.endswith("/p"):
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": "https://real.cdn/clip.mp4"})
        return _FakeResponse(status_code=200, content=b"bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealReplicateVideoClient("key")
    client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a", resolution="720p")


@pytest.mark.parametrize(
    "output,expected_url",
    [
        ("https://real.cdn/clip.mp4", "https://real.cdn/clip.mp4"),
        (["https://real.cdn/clip.mp4"], "https://real.cdn/clip.mp4"),
        ({"video": "https://real.cdn/clip.mp4"}, "https://real.cdn/clip.mp4"),
        ({"url": "https://real.cdn/clip.mp4"}, "https://real.cdn/clip.mp4"),
    ],
)
def test_generate_speech_to_video_handles_every_real_output_shape(monkeypatch, output, expected_url):
    # Real gap, 2026-10-21: Replicate's own docs never document this
    # model's output shape (only a bare-string example for an unrelated
    # text model) - a test only covering one shape would miss the others
    # actually seen across Replicate's real models.
    requested_urls = []

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        requested_urls.append(url)
        if url.endswith("/p"):
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": output})
        return _FakeResponse(status_code=200, content=b"bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealReplicateVideoClient("key")
    client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")

    assert expected_url in requested_urls


def test_generate_speech_to_video_raises_on_an_unrecognized_output_shape(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": {"unexpected_key": 123}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealReplicateVideoClient("key")
    with pytest.raises(ReplicateVideoError, match="no recognizable video URL"):
        client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")


def test_generate_speech_to_video_raises_on_a_failed_prediction(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        return _FakeResponse(status_code=200, json_data={"status": "failed"}, text='{"status": "failed", "error": "real reason"}')

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealReplicateVideoClient("key")
    with pytest.raises(ReplicateVideoError, match="real reason"):
        client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")


def test_generate_speech_to_video_raises_on_a_non_2xx_submit(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=401, text="invalid token")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealReplicateVideoClient("bad-token")
    with pytest.raises(ReplicateVideoError, match="401"):
        client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")


def test_generate_speech_to_video_raises_on_a_real_timeout(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=201, json_data={"id": "p-timeout", "urls": {"get": "https://api.replicate.com/v1/predictions/p-timeout"}})

    def fake_get(url, *, headers=None, timeout=30):
        return _FakeResponse(status_code=200, json_data={"status": "processing"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monotonic_values = iter([0, 0, 10_000])
    monkeypatch.setattr("time.monotonic", lambda: next(monotonic_values))

    client = RealReplicateVideoClient("key")
    with pytest.raises(ReplicateVideoError, match="did not complete within 1200s"):
        client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")


def test_generate_speech_to_video_retries_a_real_429_and_honors_the_real_retry_after(monkeypatch):
    # Real incident, 2026-10-21: see replicate_image_client.py's own
    # mirrored test - the identical real low-credit 429 throttle applies
    # to this endpoint, same account, same real risk from a tight loop
    # of successive submissions.
    post_calls = []
    sleep_calls = []

    def fake_post(url, *, headers, json, timeout):
        post_calls.append(1)
        if len(post_calls) == 1:
            return _FakeResponse(status_code=429, json_data={"detail": "throttled", "retry_after": 7})
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        if url.endswith("/p"):
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": "https://real.cdn/clip.mp4"})
        return _FakeResponse(status_code=200, content=b"real-mp4-bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    client = RealReplicateVideoClient("key")
    data = client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")

    assert data == b"real-mp4-bytes"
    assert len(post_calls) == 2
    assert sleep_calls == [7]


def test_generate_speech_to_video_gives_up_after_max_retries_on_persistent_429s(monkeypatch):
    post_calls = []

    def fake_post(url, *, headers, json, timeout):
        post_calls.append(1)
        return _FakeResponse(status_code=429, json_data={"detail": "throttled", "retry_after": 1})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)

    client = RealReplicateVideoClient("key")
    with pytest.raises(ReplicateVideoError, match="429"):
        client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")

    assert len(post_calls) == 5


def test_generate_speech_to_video_paces_itself_through_the_shared_submit_pacer(monkeypatch):
    # Real incident, 2026-10-21: see test_replicate_image_client.py's own
    # mirrored test - the identical real throttle applies to this
    # endpoint, same account, same real shared rate-limit bucket.
    from src.integrations.replicate_rate_limiter import shared_submit_pacer

    calls = []
    monkeypatch.setattr(shared_submit_pacer, "wait_for_slot", lambda: calls.append(1))

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        if url.endswith("/p"):
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": "https://real.cdn/clip.mp4"})
        return _FakeResponse(status_code=200, content=b"bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealReplicateVideoClient("key")
    client.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")

    assert calls == [1]


def test_mock_replicate_video_client_records_calls_and_returns_scripted_bytes():
    mock = MockReplicateVideoClient(scripted_bytes=b"scripted-clip")

    data = mock.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")

    assert data == b"scripted-clip"
    assert mock.calls == [{"prompt": "p", "image_data_uri": "i", "audio_data_uri": "a", "resolution": "480p"}]


def test_mock_replicate_video_client_fails_at_the_scripted_call_index():
    mock = MockReplicateVideoClient(fail_at_calls=frozenset({1}))
    mock.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")

    with pytest.raises(ReplicateVideoError, match="mock failure at call 1"):
        mock.generate_speech_to_video(prompt="p", image_data_uri="i", audio_data_uri="a")
