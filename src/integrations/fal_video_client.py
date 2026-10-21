"""fal.ai's queue REST API for fal-ai/wan/v2.2-14b/speech-to-video - the
real lip-synced-dialogue model (one reference photo + one audio clip + a
short action prompt -> a clip of that face speaking that audio), picked
to replace the old silent-T2V-plus-narration pipeline because the owner's
actual ask was real character dialogue, not a narrated documentary.

Contract verified live against fal's own docs 2026-10-18 (see
docs/SOURCES.md "fal.ai Wan 2.2 Speech-to-Video real contract"):

- POST https://queue.fal.run/fal-ai/wan/v2.2-14b/speech-to-video with
  header "Authorization: Key $FAL_KEY" and a JSON body of
  {prompt, image_url, audio_url, resolution} returns
  {"request_id", "status_url", "response_url", "cancel_url"}.
  image_url/audio_url accept a plain data: URI, so a local reference
  photo and a local TTS clip can be sent directly with no separate
  upload step.
- GET <status_url> returns {"status": "IN_QUEUE"|"IN_PROGRESS"|"COMPLETED", ...}
  - polled until COMPLETED (or the job errors).
- GET <response_url> returns the model's own output shape once COMPLETED;
  for this model that is {"video": {"url": ...}, ...} - the real video
  bytes are fetched from that url with a second GET.

Pricing is per second of OUTPUT video, billed at a fixed 16fps regardless
of the real fps requested: $0.10/s at 480p, $0.15/s at 580p, $0.20/s at
720p (fal.ai's own model page, see docs/SOURCES.md). PRICE_PER_SECOND_USD
below exists so DialoguePipeline can report real cost without an extra
billing call.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

ENDPOINT_ID = "fal-ai/wan/v2.2-14b/speech-to-video"
QUEUE_BASE_URL = "https://queue.fal.run"

# Real incident, 2026-10-20: two separate real jobs against this exact
# endpoint both timed out at the previous 600s cap ("fal.ai job did not
# complete within 600s"), back to back. No third-party source gives fal's
# own typical time for this model, but the closest real figure found (a
# different host's own Wan 2.2 speech-to-video deployment, see
# docs/SOURCES.md "fal.ai job timeout - real incident") reports a MEDIAN
# of ~412s for a successful run - meaning the old 600s cap sat barely
# above the midpoint of the real distribution, so a routine (not rare)
# slower-than-median run would fail outright. Raised to a real margin
# above that median rather than guessed.
POLL_TIMEOUT_S = 1200

PRICE_PER_SECOND_USD = {
    "480p": 0.10,
    "580p": 0.15,
    "720p": 0.20,
}

# Real invoice evidence, 2026-10-21 (see docs/SOURCES.md "fal.ai billed the
# 720p rate on every real call, despite every request asking for 480p"):
# the owner's own fal.ai Usage & Billing page showed this exact endpoint
# billed at precisely $0.20/video-second - the 720p row in the table
# above, never the 480p row - on every real line this project has ever
# produced, even though generate_speech_to_video() has sent
# resolution="480p" since its very first commit (confirmed via git log:
# 0b80a95). Real numbers: 117.91s billed at $0.20/s = $23.58125, matching
# the real "Total Cost" fal.ai showed almost to the cent.
#
# Until fal.ai confirms or fixes why, this project's own real-spend
# tracking (the pre-Approve estimate and the hard MAX_EPISODE_COST_USD
# ceiling) must use the rate fal.ai is REALLY charging, not the rate its
# docs promise - a ceiling computed at the hopeful $0.10/s rate is not a
# real $10 ceiling if reality is $0.20/s, it is a real $20 ceiling with a
# $10 label on it. This does not change what resolution is requested
# (still "480p" - still the cheaper, valid, documented choice; sending
# something else is not known to change what fal.ai actually does here).
REAL_BILLED_PRICE_PER_SECOND_USD = PRICE_PER_SECOND_USD["720p"]


class FalVideoError(Exception):
    pass


class FalVideoClient(ABC):
    @abstractmethod
    def generate_speech_to_video(
        self, *, prompt: str, image_data_uri: str, audio_data_uri: str, resolution: str = "480p"
    ) -> bytes:
        """Submits one real speech-to-video job, blocks until it completes,
        and returns the real mp4 bytes. Raises FalVideoError on a failed
        job or an unexpected response shape."""


class RealFalVideoClient(FalVideoClient):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_speech_to_video(
        self, *, prompt: str, image_data_uri: str, audio_data_uri: str, resolution: str = "480p"
    ) -> bytes:
        import httpx

        headers = {"Authorization": f"Key {self._api_key}", "Content-Type": "application/json"}
        submit = httpx.post(
            f"{QUEUE_BASE_URL}/{ENDPOINT_ID}",
            headers=headers,
            json={"prompt": prompt, "image_url": image_data_uri, "audio_url": audio_data_uri, "resolution": resolution},
            timeout=60,
        )
        if submit.status_code >= 400:
            raise FalVideoError(f"fal.ai submit failed ({submit.status_code}): {submit.text[:500]}")
        payload = submit.json()
        status_url = payload["status_url"]
        response_url = payload["response_url"]
        cancel_url = payload.get("cancel_url")

        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            status_resp = httpx.get(status_url, headers=headers, timeout=30)
            if status_resp.status_code >= 400:
                # Real finding, 2026-10-19: the first real run past the
                # submit step hit a real 422 here, and httpx's default
                # raise_for_status() message carries only the status line
                # and URL - never the response body, which is the one
                # thing that could have said WHY fal.ai rejected it. Every
                # real failure point in this client now reports the real
                # body, matching the submit error below, so a repeat
                # failure is diagnosable instead of another guess.
                raise FalVideoError(f"fal.ai status check failed ({status_resp.status_code}): {status_resp.text[:500]}")
            status = status_resp.json().get("status")
            if status == "COMPLETED":
                break
            if status in ("ERROR", "FAILED"):
                raise FalVideoError(f"fal.ai job failed: {status_resp.text[:500]}")
            time.sleep(3)
        else:
            # Real risk found while fixing the 2026-10-20 double timeout:
            # giving up here used to just stop polling. fal.ai's own docs
            # (docs/SOURCES.md "fal.ai cancel API") say a queue-based
            # job keeps running on their side after the connection drops -
            # "only an explicit cancel API call can stop the request" -
            # so an abandoned job could still finish and be billed with
            # nothing here left to collect the video it paid for. PUT to
            # the real cancel_url fal.ai gave at submit time at least
            # removes a still-queued job outright (200, never runs) and
            # best-effort signals an in-progress one to stop; its own
            # outcome is never allowed to hide the real timeout below.
            if cancel_url:
                try:
                    httpx.put(cancel_url, headers=headers, timeout=30)
                except Exception:
                    pass
            raise FalVideoError(
                f"fal.ai job did not complete within {POLL_TIMEOUT_S}s (request_id={payload.get('request_id')})"
            )

        result_resp = httpx.get(response_url, headers=headers, timeout=30)
        if result_resp.status_code >= 400:
            raise FalVideoError(f"fal.ai result fetch failed ({result_resp.status_code}): {result_resp.text[:500]}")
        result = result_resp.json()
        try:
            video_url = result["video"]["url"]
        except (KeyError, TypeError) as exc:
            raise FalVideoError(f"fal.ai result had no video.url: {result!r}") from exc

        video_resp = httpx.get(video_url, timeout=120)
        video_resp.raise_for_status()
        return video_resp.content


class MockFalVideoClient(FalVideoClient):
    """Scripted mp4 bytes, records every call - lets a test drive the
    whole dialogue pipeline without spending real fal.ai money.

    `fail_at_calls` lets a test simulate the real 2026-10-19 incident (a
    real line timing out partway through a real multi-line run, after
    several earlier real clips already succeeded) - a 0-indexed call
    number in that set raises FalVideoError instead of returning bytes,
    without needing a real fal.ai account to reproduce it."""

    def __init__(self, scripted_bytes: bytes = b"fake-s2v-mp4-bytes", fail_at_calls: frozenset[int] = frozenset()):
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
            raise FalVideoError(f"mock failure at call {call_index}")
        return self._scripted_bytes
