"""fal.ai's queue REST API for fal-ai/flux/dev - a real text-to-image
model, used by DialoguePipeline (`_resolve_character_photo`) to generate
one real photorealistic reference photo per (this screenplay, this
character), from that character's own `appearance` text - never a
fixed photo reused across stories, and never a generic placeholder.

Owner's instruction, 2026-10-20 (reacting to the six hand-drawn cartoon
placeholders actually appearing in a real produced video - "هذه هي
الصور التي اقصدها... لا صور متحركة" - these are the photos I mean, not
cartoon ones, showing a real photorealistic reference photo): this
client exists to generate real AI photos (not a real person's likeness
- no copyright/consent question) instead of the cartoon placeholders
that preceded it.

Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت التوليد
حسب القصة" - delete the six photos, I told you generation must be
per-story): the cartoon placeholders, and the intermediate idea of
caching one real photo per gender/age_group and reusing it across
unrelated stories, are both deleted outright - see models.py's and
character_photos.py's own docstrings. There is no fallback photo of any
kind left anywhere: if this client fails or is unavailable, that
character's lines fail and are reported, never a stand-in face.

Contract verified live against fal's own model page 2026-10-20 (WebFetch
of https://fal.ai/models/fal-ai/flux/dev, not guessed):
- Endpoint fal-ai/flux/dev, same queue.fal.run submit/status/response/
  cancel shape fal_video_client.py already documents and uses.
- Result shape is {"images": [{"url": ..., "content_type": ...}], ...} -
  NOT {"video": {"url": ...}} like the speech-to-video model.
- Pricing: $0.025 per megapixel, billed rounded up to the nearest
  megapixel - PRICE_PER_MEGAPIXEL_USD below, so a caller can report the
  real approximate cost without a separate billing call.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

ENDPOINT_ID = "fal-ai/flux/dev"
QUEUE_BASE_URL = "https://queue.fal.run"
PRICE_PER_MEGAPIXEL_USD = 0.025

# Same real-incident-driven margin as fal_video_client.py's
# POLL_TIMEOUT_S - a single still image is far faster than a video clip
# in practice, but there is no published fal.ai figure to size this from
# more precisely, so it is set generously rather than guessed tight.
POLL_TIMEOUT_S = 300


class ImageGenerationError(Exception):
    pass


class ImageClient(ABC):
    @abstractmethod
    def generate_image(self, prompt: str) -> bytes:
        """Submits one real text-to-image job, blocks until it completes,
        and returns the real image bytes. Raises ImageGenerationError on a
        failed job or an unexpected response shape."""


class RealFalImageClient(ImageClient):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_image(self, prompt: str) -> bytes:
        import httpx

        headers = {"Authorization": f"Key {self._api_key}", "Content-Type": "application/json"}
        submit = httpx.post(
            f"{QUEUE_BASE_URL}/{ENDPOINT_ID}",
            headers=headers,
            json={"prompt": prompt},
            timeout=60,
        )
        if submit.status_code >= 400:
            raise ImageGenerationError(f"fal.ai image submit failed ({submit.status_code}): {submit.text[:500]}")
        payload = submit.json()
        status_url = payload["status_url"]
        response_url = payload["response_url"]
        cancel_url = payload.get("cancel_url")

        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            status_resp = httpx.get(status_url, headers=headers, timeout=30)
            if status_resp.status_code >= 400:
                raise ImageGenerationError(
                    f"fal.ai image status check failed ({status_resp.status_code}): {status_resp.text[:500]}"
                )
            status = status_resp.json().get("status")
            if status == "COMPLETED":
                break
            if status in ("ERROR", "FAILED"):
                raise ImageGenerationError(f"fal.ai image job failed: {status_resp.text[:500]}")
            time.sleep(2)
        else:
            # Same real risk as fal_video_client.py's own timeout path:
            # an abandoned queue-based job keeps running on fal's side
            # unless explicitly cancelled.
            if cancel_url:
                try:
                    httpx.put(cancel_url, headers=headers, timeout=30)
                except Exception:
                    pass
            raise ImageGenerationError(
                f"fal.ai image job did not complete within {POLL_TIMEOUT_S}s (request_id={payload.get('request_id')})"
            )

        result_resp = httpx.get(response_url, headers=headers, timeout=30)
        if result_resp.status_code >= 400:
            raise ImageGenerationError(f"fal.ai image result fetch failed ({result_resp.status_code}): {result_resp.text[:500]}")
        result = result_resp.json()
        try:
            image_url = result["images"][0]["url"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ImageGenerationError(f"fal.ai image result had no images[0].url: {result!r}") from exc

        image_resp = httpx.get(image_url, timeout=60)
        image_resp.raise_for_status()
        return image_resp.content


class MockImageClient(ImageClient):
    def __init__(self, scripted_bytes: bytes = b"fake-portrait-jpeg-bytes"):
        self._scripted_bytes = scripted_bytes
        self.calls: list[str] = []

    def generate_image(self, prompt: str) -> bytes:
        self.calls.append(prompt)
        return self._scripted_bytes
