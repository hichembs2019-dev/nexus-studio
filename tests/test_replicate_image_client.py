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


def test_generate_image_retries_a_real_429_and_honors_the_real_retry_after(monkeypatch):
    # Real incident, 2026-10-21: a real production run failed EVERY
    # character's portrait with a real 429 from Replicate - "Your rate
    # limit for creating predictions is reduced to 6 requests per minute
    # ... while you have less than $5.0 in credit" - and the client had
    # no retry logic at all. The 429 body itself names the real wait
    # (retry_after) - this must be read and slept, not guessed.
    post_calls = []
    sleep_calls = []

    def fake_post(url, *, headers, json, timeout):
        post_calls.append(1)
        if len(post_calls) == 1:
            return _FakeResponse(status_code=429, json_data={"detail": "throttled", "retry_after": 7})
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        if url.endswith("/p"):
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": "https://real.cdn/portrait.jpg"})
        return _FakeResponse(status_code=200, content=b"real-jpeg-bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    client = RealReplicateImageClient("key")
    data = client.generate_image("p")

    assert data == b"real-jpeg-bytes"
    assert len(post_calls) == 2  # one 429, one real success - retried exactly once
    assert sleep_calls == [7]  # slept the REAL retry_after the API gave, not a guess


def test_generate_image_gives_up_after_max_retries_on_persistent_429s(monkeypatch):
    post_calls = []

    def fake_post(url, *, headers, json, timeout):
        post_calls.append(1)
        return _FakeResponse(status_code=429, json_data={"detail": "throttled", "retry_after": 1})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)

    client = RealReplicateImageClient("key")
    with pytest.raises(ReplicateImageError, match="429"):
        client.generate_image("p")

    assert len(post_calls) == 5  # MAX_SUBMIT_RETRIES - never retries forever on a genuinely broke account


def test_generate_image_paces_itself_through_the_shared_submit_pacer(monkeypatch):
    # Real incident, 2026-10-21: real portrait generations failed with a
    # real 429 from Replicate's documented low-credit/no-payment-method
    # throttle (6/min, burst of 1) even on an account with real paid
    # credit above $5 - see replicate_rate_limiter.py's own docstring.
    # Every real submission must go through the shared pacer BEFORE being
    # sent, not just react to a 429 after the fact.
    from src.integrations.replicate_rate_limiter import shared_submit_pacer

    calls = []
    monkeypatch.setattr(shared_submit_pacer, "wait_for_slot", lambda: calls.append(1))

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(status_code=201, json_data={"id": "p", "urls": {"get": "https://api.replicate.com/v1/predictions/p"}})

    def fake_get(url, *, headers=None, timeout=30):
        if url.endswith("/p"):
            return _FakeResponse(status_code=200, json_data={"status": "succeeded", "output": "https://real.cdn/portrait.jpg"})
        return _FakeResponse(status_code=200, content=b"bytes")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealReplicateImageClient("key")
    client.generate_image("p")

    assert calls == [1]


def test_mock_replicate_image_client_records_calls_and_returns_scripted_bytes():
    mock = MockReplicateImageClient(scripted_bytes=b"scripted-portrait")

    data = mock.generate_image("a portrait")

    assert data == b"scripted-portrait"
    assert mock.calls == ["a portrait"]
