"""Paces real submissions to Replicate's prediction-creation endpoint so
this app never triggers Replicate's own documented restrictive rate-limit
tier in the first place, instead of only reacting to a 429 after the fact.

Real incident, 2026-10-21: a real production run failed multiple real
character portraits with
`Replicate image submit failed (429): {"detail":"Request was throttled.
Your rate limit for creating predictions is reduced to 6 requests per
minute with a burst of 1 requests while you have less than $5.0 in
credit. ..."}` - on a real account that, at the time the owner checked it,
showed $8.42 of real, paid credit remaining (a real $10.00 purchase had
succeeded). Checked against Replicate's own docs
(https://replicate.com/docs/topics/predictions/rate-limits): the exact
numbers in the error (6/minute, burst of 1) match the tier they document
for accounts "granted credit" with "no payment method on file" - but
their own billing docs also state a payment method is REQUIRED before any
credit purchase can even be processed, which the real $10 purchase having
succeeded already proves happened. A second, independently verified real
case (a different user, $9.56 balance, identical error text) confirms
this is not unique to this account. No official Replicate source resolves
why the message fires above $5 real credit - contacting Replicate support
directly (https://replicate.com/support) is the only path to a definitive,
account-specific answer, and that is the owner's call to make, not
something to keep guessing at here.

Rather than keep guessing at Replicate's internal reason, this paces
every real submission (video and image alike - they share one real
Replicate account and therefore one real rate-limit bucket) to never
exceed the WORST tier Replicate documents, so the app cannot trip this
specific throttle regardless of why Replicate applies it to this
account. This is a NEW, separate layer from `_post_with_retry()`'s own
429-retry logic in replicate_image_client.py/replicate_video_client.py -
that logic still honors whatever `retry_after` Replicate's own response
states for a single submission's own retries; this module only paces the
gap BETWEEN separate logical submissions (e.g. one character's portrait
and the next), which is new behavior that did not exist before.
"""
from __future__ import annotations

import threading
import time

# Replicate's own documented worst-case restrictive tier:
# "you'll also be rate limited to 1 request per second with a maximum of
# 6 requests per minute" (https://replicate.com/docs/topics/predictions/
# rate-limits). 60s / 6 requests = 10s - the minimum gap that can never
# exceed that stated maximum, derived directly from their own numbers,
# not guessed.
MIN_SECONDS_BETWEEN_SUBMITS = 10.0


class SubmitPacer:
    def __init__(self, min_interval_s: float = MIN_SECONDS_BETWEEN_SUBMITS):
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_submit_monotonic: float | None = None

    def wait_for_slot(self) -> None:
        """Blocks (real time.sleep) only long enough that at least
        `min_interval_s` has passed since the last real submission this
        pacer allowed through. The very first call never waits."""
        with self._lock:
            now = time.monotonic()
            if self._last_submit_monotonic is not None:
                remaining = self._min_interval_s - (now - self._last_submit_monotonic)
                if remaining > 0:
                    time.sleep(remaining)
                    now = time.monotonic()
            self._last_submit_monotonic = now

    def reset(self) -> None:
        """Test-only: clears pacing state so a test does not inherit
        timing from whatever ran before it in the same process."""
        with self._lock:
            self._last_submit_monotonic = None


# One real, shared instance - video and image submissions hit the same
# real Replicate account and therefore the same real rate-limit bucket,
# so pacing state must be shared between replicate_video_client.py and
# replicate_image_client.py, never tracked independently by each.
shared_submit_pacer = SubmitPacer()
