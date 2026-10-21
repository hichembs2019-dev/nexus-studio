from __future__ import annotations

from src.integrations.replicate_rate_limiter import MIN_SECONDS_BETWEEN_SUBMITS, SubmitPacer


def test_the_very_first_call_never_waits(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("time.monotonic", lambda: 100.0)

    pacer = SubmitPacer(min_interval_s=10.0)
    pacer.wait_for_slot()

    assert sleep_calls == []


def test_a_second_call_inside_the_window_sleeps_exactly_the_real_remaining_gap(monkeypatch):
    # Real requirement: the wait is computed from the REAL elapsed time
    # since the last submission, not a fixed guess - 3s have really
    # passed out of the required 10s, so exactly 7s remain.
    sleep_calls = []
    monotonic_values = iter([100.0, 103.0, 103.0])
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("time.monotonic", lambda: next(monotonic_values))

    pacer = SubmitPacer(min_interval_s=10.0)
    pacer.wait_for_slot()
    pacer.wait_for_slot()

    assert sleep_calls == [7.0]


def test_a_call_after_the_window_has_already_passed_never_waits(monkeypatch):
    monotonic_values = iter([100.0, 115.0])
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("time.monotonic", lambda: next(monotonic_values))

    pacer = SubmitPacer(min_interval_s=10.0)
    pacer.wait_for_slot()
    pacer.wait_for_slot()

    assert sleep_calls == []


def test_reset_actually_clears_state_not_just_pretends_to(monkeypatch):
    # Mutation check built into the test itself: both calls read the
    # SAME monotonic value (100.0) - without a real reset, the second
    # call would see elapsed=0 and sleep the full 10s. Only a reset that
    # genuinely clears _last_submit_monotonic avoids that.
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("time.monotonic", lambda: 100.0)

    pacer = SubmitPacer(min_interval_s=10.0)
    pacer.wait_for_slot()
    pacer.reset()
    pacer.wait_for_slot()

    assert sleep_calls == []


def test_min_seconds_between_submits_matches_replicates_own_documented_worst_tier():
    # Real, documented worst-case tier (replicate.com/docs/topics/
    # predictions/rate-limits): "a maximum of 6 requests per minute".
    # 60 / 6 = 10 - derived directly from their own stated number, not a
    # separately guessed constant that could silently drift from it.
    assert MIN_SECONDS_BETWEEN_SUBMITS == 60.0 / 6
