from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _reset_replicate_submit_pacer():
    # Real risk: shared_submit_pacer (replicate_rate_limiter.py) is a
    # module-level singleton so video and image submissions share one
    # real pacing clock, the same way they share one real Replicate
    # rate-limit bucket. Left un-reset between tests, a test running
    # late in the suite would inherit timing from whatever ran before
    # it and could trigger a real (or monkeypatched-away but still
    # misleading) wait that has nothing to do with that test itself.
    from src.integrations.replicate_rate_limiter import shared_submit_pacer

    shared_submit_pacer.reset()
    yield
    shared_submit_pacer.reset()
