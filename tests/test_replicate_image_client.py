from __future__ import annotations

import pytest

from src.integrations.replicate_image_client import (
    ENDPOINT,
    MockReplicateImageClient,
    ReplicateImageError,
    RealReplicateImageClient,
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


def test_generate_image_submits_polls_and_downloads_the_real_image(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        assert url == ENDPOINT
        assert headers["Authorization"] == "Bearer real-replicate-token"
        assert json == {"input": {"prompt": "a portrait"}}
        return _FakeResponse(status_code=201, json_data={"id": "p-1", "urls": {"get": "https://api.replicate.com/v1/predictions/p-1"}})

    def fake_get(url, *, headers=None, timeout=30):
        if url.endswith("/p-1"):
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": "https://real.cdn/portrait.jpg"})
        return _FakeResponse(status_code=200, content=b"real-jpeg-bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealReplicateImageClient("real-replicate-token")
    data = client.generate_image("a portrait")

    assert data == b"real-jpeg-bytes"


@pytest.mark.parametrize(
    "output,expected_url",
    [
        ("https://real.cdn/portrait.jpg", "https://real.cdn/portrait.jpg"),
        (["https://real.cdn/portrait.jpg"], "https://real.cdn/portrait.jpg"),
        ({"image": "https://real.cdn/portrait.jpg"}, "https://real.cdn/portrait.jpg"),
    ],
)
def test_generate_image_handles_every_real_output_shape(monkeypatch, output, expected_url):
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

    client = RealReplicateImageClient("key")
    client.generate_image("p")

    assert expected_url in requested_urls


def test_generate_image_raises_on_a_failed_prediction(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        return _FakeResponse(status_code=200, json_data={"status": "failed"}, text='{"status": "failed", "error": "real reason"}')

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealReplicateImageClient("key")
    with pytest.raises(ReplicateImageError, match="real reason"):
        client.generate_image("p")


def test_generate_image_raises_with_the_real_body_on_a_failed_submit(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=401, text="invalid token")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealReplicateImageClient("bad-token")
    with pytest.raises(ReplicateImageError, match="401"):
        client.generate_image("p")


def test_generate_image_raises_on_a_real_timeout(monkeypatch):
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

    client = RealReplicateImageClient("key")
    with pytest.raises(ReplicateImageError, match="did not complete within 300s"):
        client.generate_image("p")


def test_mock_replicate_image_client_records_calls_and_returns_scripted_bytes():
    mock = MockReplicateImageClient(scripted_bytes=b"scripted-portrait")

    data = mock.generate_image("a portrait")

    assert data == b"scripted-portrait"
    assert mock.calls == ["a portrait"]
