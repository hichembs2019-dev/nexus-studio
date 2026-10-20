from __future__ import annotations

import pytest

from src.integrations.image_client import ENDPOINT_ID, ImageGenerationError, MockImageClient, RealFalImageClient


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


def test_generate_image_submits_polls_and_downloads_the_real_image(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        assert url == f"https://queue.fal.run/{ENDPOINT_ID}"
        assert headers["Authorization"] == "Key real-fal-key"
        assert json == {"prompt": "a portrait"}
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
        if url.endswith("/status"):
            return _FakeResponse(status_code=200, json_data={"status": "COMPLETED"})
        if url == "https://queue.fal.run/x/requests/req-1":
            return _FakeResponse(status_code=200, json_data={"images": [{"url": "https://cdn.fal.media/x.jpg"}]})
        if url == "https://cdn.fal.media/x.jpg":
            return _FakeResponse(status_code=200, content=b"real-jpeg-bytes")
        raise AssertionError(f"unexpected GET {url}")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealFalImageClient("real-fal-key")
    data = client.generate_image("a portrait")

    assert data == b"real-jpeg-bytes"


def test_generate_image_raises_on_a_failed_job(monkeypatch):
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

    client = RealFalImageClient("key")
    with pytest.raises(ImageGenerationError, match="failed"):
        client.generate_image("p")


def test_generate_image_cancels_the_real_job_on_a_real_timeout(monkeypatch):
    # Same real lesson as fal_video_client.py's own timeout fix: fal's own
    # docs say a queue-based job keeps running after the client stops
    # polling, so giving up must call the real cancel_url, not just stop
    # waiting - otherwise an abandoned job can still finish and be billed
    # with nothing here to collect.
    cancel_calls = []

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(
            status_code=200,
            json_data={
                "request_id": "req-timeout",
                "status_url": "https://queue.fal.run/x/requests/req-timeout/status",
                "response_url": "https://queue.fal.run/x/requests/req-timeout",
                "cancel_url": "https://queue.fal.run/x/requests/req-timeout/cancel",
            },
        )

    def fake_get(url, *, headers=None, timeout=30):
        return _FakeResponse(status_code=200, json_data={"status": "IN_PROGRESS"})

    def fake_put(url, *, headers=None, timeout=30):
        cancel_calls.append(url)
        return _FakeResponse(status_code=202, json_data={"status": "CANCELLATION_REQUESTED"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "put", fake_put)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    monotonic_values = iter([0, 0, 10_000])
    monkeypatch.setattr("time.monotonic", lambda: next(monotonic_values))

    client = RealFalImageClient("key")
    with pytest.raises(ImageGenerationError, match="did not complete"):
        client.generate_image("p")

    assert cancel_calls == ["https://queue.fal.run/x/requests/req-timeout/cancel"]


def test_generate_image_raises_with_the_real_body_on_a_failed_submit(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=401, text="invalid key")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealFalImageClient("bad-key")
    with pytest.raises(ImageGenerationError, match="401"):
        client.generate_image("p")


def test_mock_image_client_records_calls_and_returns_scripted_bytes():
    mock = MockImageClient(scripted_bytes=b"scripted-portrait")

    data = mock.generate_image("a portrait")

    assert data == b"scripted-portrait"
    assert mock.calls == ["a portrait"]
