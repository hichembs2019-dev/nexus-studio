from __future__ import annotations

import pytest

from src.integrations.publishing_client import MockPublishingClient, PublishingError, RealYouTubeClient


class _FakeHttpxResponse:
    def __init__(self, json_payload=None, headers=None, status_code=200):
        self._json = json_payload
        self.headers = headers or {}
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_get_access_token_posts_the_real_refresh_grant_shape(monkeypatch):
    calls = []

    def fake_post(url, *, data=None, headers=None, json=None, params=None, timeout=None):
        calls.append((url, data))
        if url == RealYouTubeClient._TOKEN_URL:
            return _FakeHttpxResponse({"access_token": "fake-access-token", "expires_in": 3599, "token_type": "Bearer"})
        raise AssertionError(f"unexpected POST to {url}")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealYouTubeClient("client-id", "client-secret", "refresh-token")
    token = client._get_access_token()

    assert token == "fake-access-token"
    url, data = calls[0]
    assert url == "https://oauth2.googleapis.com/token"
    assert data == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "refresh_token": "refresh-token",
        "grant_type": "refresh_token",
    }


def test_get_access_token_raises_if_no_access_token_in_response(monkeypatch):
    def fake_post(url, *, data=None, headers=None, json=None, params=None, timeout=None):
        return _FakeHttpxResponse({"error": "invalid_grant"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealYouTubeClient("id", "secret", "bad-refresh-token")

    with pytest.raises(PublishingError, match="did not return an access_token"):
        client._get_access_token()


def test_upload_video_does_the_real_two_step_resumable_protocol(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video bytes")

    posts = []
    puts = []

    def fake_post(url, *, data=None, headers=None, json=None, params=None, timeout=None):
        if url == RealYouTubeClient._TOKEN_URL:
            return _FakeHttpxResponse({"access_token": "fake-access-token"})
        if url == RealYouTubeClient._UPLOAD_URL:
            posts.append((url, params, headers, json))
            return _FakeHttpxResponse(headers={"Location": "https://upload.example/session/abc123"})
        raise AssertionError(f"unexpected POST to {url}")

    def fake_put(url, *, headers=None, content=None, timeout=None):
        puts.append((url, headers, content))
        return _FakeHttpxResponse({"id": "yt-video-id-123"}, status_code=201)

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "put", fake_put)

    client = RealYouTubeClient("id", "secret", "refresh")
    video_id = client.upload_video(str(video_path), title="T", description="D", tags=["a", "b"], privacy_status="unlisted")

    assert video_id == "yt-video-id-123"

    _, params, headers, body = posts[0]
    assert params == {"uploadType": "resumable", "part": "snippet,status"}
    assert headers["Authorization"] == "Bearer fake-access-token"
    assert headers["X-Upload-Content-Length"] == str(len(b"fake video bytes"))
    assert headers["X-Upload-Content-Type"] == "video/mp4"
    assert body == {"snippet": {"title": "T", "description": "D", "tags": ["a", "b"]}, "status": {"privacyStatus": "unlisted"}}

    put_url, put_headers, put_content = puts[0]
    assert put_url == "https://upload.example/session/abc123"
    assert put_headers["Authorization"] == "Bearer fake-access-token"
    assert put_content == b"fake video bytes"


def test_upload_video_raises_if_no_location_header(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"x")

    def fake_post(url, *, data=None, headers=None, json=None, params=None, timeout=None):
        if url == RealYouTubeClient._TOKEN_URL:
            return _FakeHttpxResponse({"access_token": "tok"})
        return _FakeHttpxResponse(headers={})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealYouTubeClient("id", "secret", "refresh")

    with pytest.raises(PublishingError, match="no Location header"):
        client.upload_video(str(video_path), title="T", description="D", tags=[], privacy_status="unlisted")


def test_mock_publishing_client_scripts_video_ids_and_records_uploads():
    mock = MockPublishingClient(scripted_video_ids=["id1"])

    video_id = mock.upload_video("clip.mp4", title="T", description="D", tags=["x"], privacy_status="unlisted")

    assert video_id == "id1"
    assert mock.uploads == [{"video_path": "clip.mp4", "title": "T", "description": "D", "tags": ["x"], "privacy_status": "unlisted"}]
