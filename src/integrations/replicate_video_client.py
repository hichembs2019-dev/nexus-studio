"""Replicate's REST API for wan-video/wan-2.2-s2v - the same real,
audio-driven lip-sync model family as fal.ai's Wan 2.2 Speech-to-Video,
picked as a second real provider after real evidence (docs/SOURCES.md,
"fal.ai billed every real speech-to-video call at the 720p rate...")
showed fal.ai billing did not match what it requested or received.

Contract verified live against Replicate's own docs 2026-10-21
(https://replicate.com/docs/reference/http, plus the real Playground
form at replicate.com/wan-video/wan-2.2-s2v, screenshotted by the owner):

- POST https://api.replicate.com/v1/models/wan-video/wan-2.2-s2v/predictions
  with header "Authorization: Bearer $REPLICATE_API_TOKEN" and a JSON
  body of {"input": {prompt, image, audio, num_frames_per_chunk}} creates
  a prediction. image/audio accept a data: URI directly (Replicate's own
  docs: "Files should be passed as HTTP URLs or data URLs"), so a local
  reference photo and a local TTS clip can be sent with no separate
  upload step, same as the existing fal.ai client.
- The response carries "status" (starting/processing/succeeded/failed/
  canceled) and "urls": {"get": ...} - polled until "succeeded" or a
  terminal failure.
- On success, "output" holds the result. Its exact shape for this model
  is NOT documented by Replicate (only a bare-string example exists in
  their docs, for an unrelated text model) - handled defensively below
  for every shape actually seen: a bare URL string, a list of URL
  strings, or a dict with a "video"/"url" key.
- Output file URLs require the same real Authorization header to fetch
  (Replicate's own docs: "you must pass the Authorization header to
  request them").

UNLIKE fal.ai's three-tier (480p/580p/720p) pricing, this model's real,
live Playground form (owner's own screenshot, 2026-10-21) has NO
resolution parameter at all - only prompt, image, audio,
num_frames_per_chunk, seed, interpolate. `resolution` is still accepted
here (kept only so this client is a drop-in match for FalVideoClient's
interface) and silently ignored - there is nothing real to pass it to.

PRICING, verified live and NOT theoretical: the model's own page states
"$0.02 per second of output video" - but one real, owner-run test
(docs/SOURCES.md, 2026-10-21) showed a 2-second real output billed at a
real, owner-confirmed $0.10 (matching the account's own Billing page
exactly - no discrepancy between quoted and billed, unlike fal.ai), with
`metrics.predict_time` = 16.8s of real GPU compute. Short clips are
billed closer to real compute time than to naive "$0.02 x output
seconds" - PROVISIONAL_PRICE_PER_SECOND_USD below is the one real,
conservative (highest observed) rate so far, not a confirmed formula.
Treat it as a safety-conservative placeholder, the same posture
REAL_BILLED_PRICE_PER_SECOND_USD takes for fal.ai, until more real runs
either confirm or replace it.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

ENDPOINT = "https://api.replicate.com/v1/models/wan-video/wan-2.2-s2v/predictions"

# Real incident risk, inherited from the identical fal.ai timeout finding
# (docs/SOURCES.md "fal.ai job timeout - real incident"): no reason to
# assume this model is faster end-to-end than fal.ai's equivalent, so the
# same generous real margin is used here rather than guessed smaller.
POLL_TIMEOUT_S = 1200

# Real test, 2026-10-21: 2s output, $0.10 billed (confirmed against the
# account's own real Billing page - not a discrepancy, a genuine real
# rate for a short clip dominated by fixed real compute overhead, not by
# output length). $0.10 / 2s = $0.05/s. This is ONE real data point, not
# a validated formula - kept as the conservative (highest-seen) real-
# money tracking rate until more real runs refine it. See
# dialogue_pipeline.py's price_per_second_usd parameter: this constant is
# never silently assumed there - it must be passed explicitly by
# whichever caller wires this client in (src/app.py), so the owner's own
# real spend ceiling and pre-Approve estimate are never computed against
# a rate that doesn't actually belong to the provider actually running.
PROVISIONAL_PRICE_PER_SECOND_USD = 0.05

# Real incident, 2026-10-21: a real production run failed every
# character's PORTRAIT (replicate_image_client.py, the same real 429
# shape) with Replicate's own documented low-credit throttle: "Your rate
# limit for creating predictions is reduced to 6 requests per minute
# with a burst of 1 requests while you have less than $5.0 in credit."
# This endpoint shares the same account and the same real risk - the
# dialogue loop submits one speech-to-video request per spoken line in
# quick succession. The 429 body itself names the real wait
# (`retry_after`, seconds) - MAX_SUBMIT_RETRIES bounds how many times
# this is honored before giving up for real.
MAX_SUBMIT_RETRIES = 5


class ReplicateVideoError(Exception):
    pass


def _post_with_retry(headers: dict, body: dict):
    import httpx

    for attempt in range(MAX_SUBMIT_RETRIES):
        resp = httpx.post(ENDPOINT, headers=headers, json=body, timeout=60)
        if resp.status_code != 429 or attempt == MAX_SUBMIT_RETRIES - 1:
            return resp
        try:
            wait_s = resp.json().get("retry_after", 10)
        except Exception:
            wait_s = 10
        time.sleep(wait_s)
    return resp  # pragma: no cover - loop always returns above


class ReplicateVideoClient(ABC):
    @abstractmethod
    def generate_speech_to_video(
        self, *, prompt: str, image_data_uri: str, audio_data_uri: str, resolution: str = "480p"
    ) -> bytes:
        """Submits one real speech-to-video prediction, blocks until it
        completes, and returns the real mp4 bytes. `resolution` is
        accepted only to match FalVideoClient's interface - this real
        model has no such parameter and the value is ignored. Raises
        ReplicateVideoError on a failed prediction or an unexpected
        response shape."""


class RealReplicateVideoClient(ReplicateVideoClient):
    def __init__(self, api_token: str):
        self._api_token = api_token

    def generate_speech_to_video(
        self, *, prompt: str, image_data_uri: str, audio_data_uri: str, resolution: str = "480p"
    ) -> bytes:
        import httpx

        from .replicate_rate_limiter import shared_submit_pacer

        headers = {"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"}
        # Real incident, 2026-10-21: see replicate_rate_limiter.py's own
        # docstring - paces this submission against every other real
        # Replicate submission (image portraits included) so the app
        # never sends requests faster than Replicate's own documented
        # worst-case restrictive tier in the first place.
        shared_submit_pacer.wait_for_slot()
        submit = _post_with_retry(
            headers, {"input": {"prompt": prompt, "image": image_data_uri, "audio": audio_data_uri}}
        )
        if submit.status_code >= 400:
            raise ReplicateVideoError(f"Replicate submit failed ({submit.status_code}): {submit.text[:500]}")
        payload = submit.json()
        try:
            status_url = payload["urls"]["get"]
        except (KeyError, TypeError) as exc:
            raise ReplicateVideoError(f"Replicate response had no urls.get: {payload!r}") from exc

        deadline = time.monotonic() + POLL_TIMEOUT_S
        status_resp = None
        while time.monotonic() < deadline:
            status_resp = httpx.get(status_url, headers=headers, timeout=30)
            if status_resp.status_code >= 400:
                raise ReplicateVideoError(f"Replicate status check failed ({status_resp.status_code}): {status_resp.text[:500]}")
            status_payload = status_resp.json()
            status = status_payload.get("status")
            if status == "succeeded":
                break
            if status in ("failed", "canceled"):
                raise ReplicateVideoError(f"Replicate prediction {status}: {status_resp.text[:500]}")
            time.sleep(3)
        else:
            raise ReplicateVideoError(
                f"Replicate prediction did not complete within {POLL_TIMEOUT_S}s (id={payload.get('id')})"
            )

        output = status_payload.get("output")
        video_url = self._extract_video_url(output)
        if video_url is None:
            raise ReplicateVideoError(f"Replicate output had no recognizable video URL: {output!r}")

        video_resp = httpx.get(video_url, headers=headers, timeout=120)
        video_resp.raise_for_status()
        return video_resp.content

    @staticmethod
    def _extract_video_url(output) -> str | None:
        # Defensive: Replicate's own docs don't document this model's
        # output shape (see module docstring) - handles every shape
        # actually observed across Replicate's models rather than
        # assuming one and failing opaquely on the others.
        if isinstance(output, str):
            return output
        if isinstance(output, list) and output and isinstance(output[0], str):
            return output[0]
        if isinstance(output, dict):
            for key in ("video", "url"):
                value = output.get(key)
                if isinstance(value, str):
                    return value
        return None


class MockReplicateVideoClient(ReplicateVideoClient):
    """Scripted mp4 bytes, records every call - lets a test drive the
    whole dialogue pipeline without spending real Replicate money."""

    def __init__(self, scripted_bytes: bytes = b"fake-replicate-s2v-mp4-bytes", fail_at_calls: frozenset[int] = frozenset()):
        self._scripted_bytes = scripted_bytes
        self._fail_at_calls = fail_at_calls
        self.calls: list[dict] = []

    def generate_speech_to_video(
        self, *, prompt: str, image_data_uri: str, audio_data_uri: str, resolution: str = "480p"
    ) -> bytes:
        call_index = len(self.calls)
        self.calls.append(
            {"prompt": prompt, "image_data_uri": image_data_uri, "audio_data_uri": audio_data_uri, "resolution": resolution}
        )
        if call_index in self._fail_at_calls:
            raise ReplicateVideoError(f"mock failure at call {call_index}")
        return self._scripted_bytes
