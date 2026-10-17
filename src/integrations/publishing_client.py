"""One video, one platform. YouTube's Real client is built from Google's
own official docs (verified 2026-10-17, see docs/SOURCES.md "YouTube Data
API v3 upload" and "Google OAuth2 refresh token") - the wire shapes are
real, but no authenticated call has been made live yet: that needs the
owner's own OAuth consent (client_id/client_secret from a Google Cloud
project, a refresh_token obtained once through a browser consent screen).
Until those real credentials exist, build_app falls back to the mock, same
convention as nexus-commerce's CJ Dropshipping client before it had real
keys (see that project's docs/UNVERIFIED.md).

Every other platform (TikTok, X, Instagram, Snapchat, Twitch, Kick) has NO
client here at all yet - not even an unverified stub - because guessing a
request shape nobody has read the docs for is exactly what this project
has committed not to do. See docs/SOURCES.md "Publishing - platforms not
yet built" for what each one actually needs before it can be built for
real.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class PublishingError(Exception):
    pass


class PublishingClient(ABC):
    @abstractmethod
    def upload_video(self, video_path: str, *, title: str, description: str, tags: list[str], privacy_status: str) -> str:
        """Uploads the real file at video_path, returns the platform's own video id."""


class RealYouTubeClient(PublishingClient):
    """YouTube Data API v3, resumable upload protocol - verified against
    Google's own docs 2026-10-17 (see docs/SOURCES.md), not yet exercised
    with a real access token.

    OAuth: client_id/client_secret are a Google Cloud project's OAuth 2.0
    credentials (console.cloud.google.com); refresh_token is obtained once
    through a browser consent screen authorizing the
    youtube.upload scope - this client only does the refresh->access_token
    exchange, never the initial consent flow.
    """

    _TOKEN_URL = "https://oauth2.googleapis.com/token"
    _UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token

    def _get_access_token(self) -> str:
        import httpx

        response = httpx.post(
            self._TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if "access_token" not in payload:
            raise PublishingError(f"YouTube token refresh did not return an access_token: {payload}")
        return payload["access_token"]

    def upload_video(self, video_path: str, *, title: str, description: str, tags: list[str], privacy_status: str = "unlisted") -> str:
        import os

        import httpx

        access_token = self._get_access_token()
        file_size = os.path.getsize(video_path)

        start = httpx.post(
            self._UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(file_size),
                "X-Upload-Content-Type": "video/mp4",
            },
            json={"snippet": {"title": title, "description": description, "tags": tags}, "status": {"privacyStatus": privacy_status}},
            timeout=30,
        )
        start.raise_for_status()
        session_uri = start.headers.get("Location")
        if not session_uri:
            raise PublishingError(f"YouTube resumable session had no Location header: {dict(start.headers)}")

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        upload = httpx.put(
            session_uri,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Length": str(file_size),
                "Content-Type": "video/mp4",
            },
            content=video_bytes,
            timeout=600,
        )
        upload.raise_for_status()
        video = upload.json()
        if "id" not in video:
            raise PublishingError(f"YouTube upload response had no video id: {video}")
        return video["id"]


class MockPublishingClient(PublishingClient):
    def __init__(self, scripted_video_ids: list[str] | None = None):
        self._video_ids = list(scripted_video_ids or ["mock-video-id"])
        self.uploads: list[dict] = []

    def upload_video(self, video_path: str, *, title: str, description: str, tags: list[str], privacy_status: str) -> str:
        self.uploads.append(
            {"video_path": video_path, "title": title, "description": description, "tags": tags, "privacy_status": privacy_status}
        )
        if not self._video_ids:
            raise AssertionError("MockPublishingClient ran out of scripted video ids")
        return self._video_ids.pop(0)
