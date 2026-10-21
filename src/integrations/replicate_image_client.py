"""Replicate's REST API for black-forest-labs/flux-dev - a real
text-to-image model, used the same way fal-ai/flux/dev was: one real
photorealistic reference photo per (this screenplay, this character),
from the character's own `appearance` text.

Owner's standing instruction, 2026-10-21 ("لا تضع السراق ... احذفهم من
الكود" - don't put the thieves [fal.ai] in, remove them from the code):
after fal.ai's own real billing could not be trusted for video (see
docs/SOURCES.md, the 720p/480p finding), the owner wants fal.ai out of
the active path entirely - not just for video, for photos too. This
client replaces RealFalImageClient the same way RealReplicateVideoClient
replaced RealFalVideoClient.

Contract: POST https://api.replicate.com/v1/models/black-forest-labs/
flux-dev/predictions, same real auth/poll/output shape already verified
for wan-video/wan-2.2-s2v (replicate_video_client.py) against Replicate's
own docs (https://replicate.com/docs/reference/http) - that verification
covers the platform-wide contract (auth header, urls.get polling, output
field), not this specific model's input schema.

HONESTLY UNVERIFIED, 2026-10-21: unlike the video model (whose input
schema the owner's own Playground screenshots confirmed field-by-field)
and unlike its real $0.10/2s price (owner's own real paid test), this
model's exact input field names and its real live per-image price could
NOT be confirmed here - WebFetch against both replicate.com/black-forest-
labs/flux-dev and its /api page returned no pricing and no input schema
(client-side-rendered content a fetch tool cannot see), and the public
model-metadata API requires an authenticated token this session does not
hold. `prompt` below is the one field name actually used in Replicate's
own blog posts about this model. PRICE_PER_IMAGE_USD_UNVERIFIED is
carried over from fal.ai's own $0.025/megapixel figure as a placeholder
only - NOT a confirmed Replicate price the way the video rate is. Treat
any cost shown for a portrait as approximate until a real owner-run test
(the same kind already done for video) confirms it.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

ENDPOINT = "https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions"
POLL_TIMEOUT_S = 300

# See module docstring: NOT a confirmed real Replicate price - carried
# over from fal.ai's own figure as a placeholder only, pending a real
# owner-run test the same way the video price was confirmed.
PRICE_PER_IMAGE_USD_UNVERIFIED = 0.025

# Real incident, 2026-10-21: a real production run failed EVERY
# character's portrait with a real 429 from Replicate's own API:
# "Your rate limit for creating predictions is reduced to 6 requests
# per minute with a burst of 1 requests while you have less than $5.0
# in credit." This project's own pre-pass loop (dialogue_pipeline.py)
# submits one portrait request per speaking character in quick
# succession - exactly the pattern this real, documented low-credit
# throttle exists to reject. The 429 response body itself names the
# real wait (`retry_after`, seconds) - MAX_SUBMIT_RETRIES bounds how
# many times this is honored before giving up for real, rather than
# retrying forever on a account that may simply be out of credit.
MAX_SUBMIT_RETRIES = 5


class ReplicateImageError(Exception):
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


class ReplicateImageClient(ABC):
    @abstractmethod
    def generate_image(self, prompt: str) -> bytes:
        """Submits one real text-to-image prediction, blocks until it
        completes, and returns the real image bytes. Raises
        ReplicateImageError on a failed prediction or an unexpected
        response shape."""


class RealReplicateImageClient(ReplicateImageClient):
    def __init__(self, api_token: str):
        self._api_token = api_token

    def generate_image(self, prompt: str) -> bytes:
        import httpx

        headers = {"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"}
        submit = _post_with_retry(headers, {"input": {"prompt": prompt}})
        if submit.status_code >= 400:
            raise ReplicateImageError(f"Replicate image submit failed ({submit.status_code}): {submit.text[:500]}")
        payload = submit.json()
        try:
            status_url = payload["urls"]["get"]
        except (KeyError, TypeError) as exc:
            raise ReplicateImageError(f"Replicate image response had no urls.get: {payload!r}") from exc

        deadline = time.monotonic() + POLL_TIMEOUT_S
        status_payload = None
        while time.monotonic() < deadline:
            status_resp = httpx.get(status_url, headers=headers, timeout=30)
            if status_resp.status_code >= 400:
                raise ReplicateImageError(f"Replicate image status check failed ({status_resp.status_code}): {status_resp.text[:500]}")
            status_payload = status_resp.json()
            status = status_payload.get("status")
            if status == "succeeded":
                break
            if status in ("failed", "canceled"):
                raise ReplicateImageError(f"Replicate image prediction {status}: {status_resp.text[:500]}")
            time.sleep(2)
        else:
            raise ReplicateImageError(
                f"Replicate image prediction did not complete within {POLL_TIMEOUT_S}s (id={payload.get('id')})"
            )

        output = status_payload.get("output")
        image_url = self._extract_image_url(output)
        if image_url is None:
            raise ReplicateImageError(f"Replicate image output had no recognizable URL: {output!r}")

        image_resp = httpx.get(image_url, headers=headers, timeout=60)
        image_resp.raise_for_status()
        return image_resp.content

    @staticmethod
    def _extract_image_url(output) -> str | None:
        # Defensive, same reasoning as replicate_video_client.py's own
        # _extract_video_url - this model's exact output shape could not
        # be confirmed live (see module docstring), so every shape
        # plausibly used across Replicate's real models is handled.
        if isinstance(output, str):
            return output
        if isinstance(output, list) and output and isinstance(output[0], str):
            return output[0]
        if isinstance(output, dict):
            for key in ("image", "url"):
                value = output.get(key)
                if isinstance(value, str):
                    return value
        return None


class MockReplicateImageClient(ReplicateImageClient):
    def __init__(self, scripted_bytes: bytes = b"fake-replicate-portrait-jpeg-bytes"):
        self._scripted_bytes = scripted_bytes
        self.calls: list[str] = []

    def generate_image(self, prompt: str) -> bytes:
        self.calls.append(prompt)
        return self._scripted_bytes
