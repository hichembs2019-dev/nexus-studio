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


def _post_with_retry(headers: dict, body: dict, endpoint: str = ENDPOINT):
    import httpx

    for attempt in range(MAX_SUBMIT_RETRIES):
        resp = httpx.post(endpoint, headers=headers, json=body, timeout=60)
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
        return _run_prediction(self._api_token, ENDPOINT, {"prompt": prompt})


def _run_prediction(api_token: str, endpoint: str, model_input: dict) -> bytes:
    """Submit -> poll urls.get -> download, shared by every Replicate image
    model this project calls (same real contract for all of them)."""
    import httpx

    from .replicate_rate_limiter import shared_submit_pacer

    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    # Real incident, 2026-10-21: see replicate_rate_limiter.py's own
    # docstring - paces this submission against every other real
    # Replicate submission (video included) so the app never sends
    # requests faster than Replicate's own documented worst-case
    # restrictive tier in the first place.
    shared_submit_pacer.wait_for_slot()
    submit = _post_with_retry(headers, {"input": model_input}, endpoint)
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
    image_url = _extract_image_url(output)
    if image_url is None:
        raise ReplicateImageError(f"Replicate image output had no recognizable URL: {output!r}")

    image_resp = httpx.get(image_url, headers=headers, timeout=60)
    image_resp.raise_for_status()
    return image_resp.content


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


# --- black-forest-labs/flux-2-dev: one shot still per line, face kept ---
#
# Owner's choice, 2026-10-23 ("الخيار أ"): one reference still per shot,
# generated FROM the character's own portrait so the face stays the same
# person across very different framings. Input fields read from
# Replicate's own model files, 2026-10-23
# (replicate.com/black-forest-labs/flux-2-dev/llms.txt and /api/schema),
# not guessed: prompt (required), input_images (array of URIs, max 5),
# aspect_ratio (enum incl. "16:9"), output_format (enum webp/jpg/png).
# width/height are documented for text-to-image mode only, so the
# cinematic frame is requested with aspect_ratio="16:9" and the real
# output size is measured by the caller, never assumed.
FLUX2_DEV_ENDPOINT = "https://api.replicate.com/v1/models/black-forest-labs/flux-2-dev/predictions"
FLUX2_DEV_MAX_INPUT_IMAGES = 5
SHOT_ASPECT_RATIO = "16:9"

# NOT a confirmed live price. Replicate's own blog (2025-11-25, "Run
# FLUX.2 on Replicate"): "$0.012 per input and output megapixels".
# Third-party listings in 2026 show $0.014. The higher figure is used -
# same conservative posture as PROVISIONAL_PRICE_PER_SECOND_USD - until a
# real owner-run test confirms the billed amount.
FLUX2_DEV_PRICE_PER_MEGAPIXEL_USD_UNVERIFIED = 0.014


def flux2_dev_cost_usd(pixel_counts: list[int], price_per_megapixel_usd: float = FLUX2_DEV_PRICE_PER_MEGAPIXEL_USD_UNVERIFIED) -> float:
    """Cost of one prediction from the real pixel count of every input
    image and of the output image. Each image is rounded UP to whole
    megapixels (1MP = 1,000,000 px) - a deliberate overestimate, since the
    exact rounding rule could not be confirmed from Replicate itself."""
    return sum(max(1, -(-px // 1_000_000)) for px in pixel_counts) * price_per_megapixel_usd


# Pre-Approve estimate only (the real cost is measured after generation):
# the real image sizes are unknown before anything is generated, so the
# worst case is assumed - two reference portraits plus the output, each
# rounded up to 2MP.
ESTIMATED_SHOT_MEGAPIXELS = 6


def estimated_shot_image_usd(price_per_megapixel_usd: float = FLUX2_DEV_PRICE_PER_MEGAPIXEL_USD_UNVERIFIED) -> float:
    return ESTIMATED_SHOT_MEGAPIXELS * price_per_megapixel_usd


class ShotImageClient(ABC):
    @abstractmethod
    def generate_shot_image(self, prompt: str, reference_images: list[tuple[bytes, str]]) -> bytes:
        """One real image-editing prediction: `reference_images` is a list
        of (bytes, mime) - the speaking character's portrait first.
        Returns the real image bytes. Raises ReplicateImageError."""


class RealFlux2ShotImageClient(ShotImageClient):
    def __init__(self, api_token: str):
        self._api_token = api_token

    def generate_shot_image(self, prompt: str, reference_images: list[tuple[bytes, str]]) -> bytes:
        import base64

        if not reference_images:
            raise ReplicateImageError("flux-2-dev shot needs at least one reference image (the character's portrait)")
        if len(reference_images) > FLUX2_DEV_MAX_INPUT_IMAGES:
            raise ReplicateImageError(f"flux-2-dev accepts at most {FLUX2_DEV_MAX_INPUT_IMAGES} input images")
        model_input = {
            "prompt": prompt,
            # Replicate's own HTTP docs: "Files should be passed as HTTP
            # URLs or data URLs" (docs/SOURCES.md, the same proven path the
            # video client already uses).
            "input_images": [
                f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}" for data, mime in reference_images
            ],
            "aspect_ratio": SHOT_ASPECT_RATIO,
            "output_format": "jpg",
        }
        return _run_prediction(self._api_token, FLUX2_DEV_ENDPOINT, model_input)


class MockShotImageClient(ShotImageClient):
    def __init__(self, scripted_bytes: bytes = b"fake-flux2-shot-jpeg-bytes", fail: bool = False):
        self._scripted_bytes = scripted_bytes
        self._fail = fail
        self.calls: list[tuple[str, list[tuple[bytes, str]]]] = []

    def generate_shot_image(self, prompt: str, reference_images: list[tuple[bytes, str]]) -> bytes:
        self.calls.append((prompt, reference_images))
        if self._fail:
            raise ReplicateImageError("mock flux-2-dev failure")
        return self._scripted_bytes
