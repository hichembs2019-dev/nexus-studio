from __future__ import annotations

import json
import urllib.error

import pytest

from scripts.pod_idle_watchdog import (
    IdleTracker,
    _read_credential,
    is_comfyui_idle,
    main,
    stop_pod,
)


# --- IdleTracker (pure decision logic) ---

def test_tracker_never_stops_during_the_grace_period_even_if_idle():
    tracker = IdleTracker(idle_timeout_s=100, grace_period_s=500, started_at=0.0)
    assert tracker.observe(True, now=400.0) is False


def test_tracker_stops_once_idle_for_the_full_timeout_after_grace():
    tracker = IdleTracker(idle_timeout_s=100, grace_period_s=0, started_at=0.0)
    assert tracker.observe(True, now=50.0) is False
    assert tracker.observe(True, now=150.0) is True


def test_tracker_resets_only_on_confirmed_real_work():
    tracker = IdleTracker(idle_timeout_s=100, grace_period_s=0, started_at=0.0)
    tracker.observe(True, now=90.0)
    assert tracker.observe(False, now=95.0) is False
    assert tracker.observe(True, now=150.0) is False  # only 55s since the reset at 95
    assert tracker.observe(True, now=196.0) is True  # 101s since the reset


def test_tracker_treats_an_unreachable_queue_the_same_as_idle():
    # Mutation check: a version that treated None (ComfyUI unreachable)
    # as "assume busy, never stop" would never trip this - and that is
    # precisely the owner's real incident shape (a pod with nothing
    # actually working on it, still billing, undetected).
    tracker = IdleTracker(idle_timeout_s=100, grace_period_s=0, started_at=0.0)
    assert tracker.observe(None, now=150.0) is True


# --- is_comfyui_idle ---

class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_is_comfyui_idle_true_when_the_real_queue_is_empty(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeHTTPResponse({"queue_running": [], "queue_pending": []}),
    )
    assert is_comfyui_idle() is True


def test_is_comfyui_idle_false_when_real_work_is_running(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeHTTPResponse({"queue_running": [["job-1", {}]], "queue_pending": []}),
    )
    assert is_comfyui_idle() is False


def test_is_comfyui_idle_false_when_real_work_is_only_pending(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeHTTPResponse({"queue_running": [], "queue_pending": [["job-2", {}]]}),
    )
    assert is_comfyui_idle() is False


def test_is_comfyui_idle_returns_none_when_comfyui_is_unreachable(monkeypatch):
    def raise_urlerror(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", raise_urlerror)
    assert is_comfyui_idle() is None


# --- stop_pod ---

def test_stop_pod_posts_the_real_v2_action_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode())
        captured["auth"] = req.get_header("Authorization")
        return _FakeHTTPResponse({})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    stop_pod("real-key", "pod-123")

    assert captured["url"] == "https://api.runpod.io/v2/pods/pod-123/action"
    assert captured["data"] == {"action": "stop"}
    assert captured["auth"] == "Bearer real-key"


# --- credentials ---

def test_read_credential_prefers_the_env_var_over_the_file(tmp_path, monkeypatch):
    cred_dir = tmp_path / "creds"
    cred_dir.mkdir()
    (cred_dir / "api_key").write_text("file-value")
    monkeypatch.setenv("WATCHDOG_CRED_DIR", str(cred_dir))
    monkeypatch.setenv("RUNPOD_API_KEY", "env-value")

    assert _read_credential("api_key", "RUNPOD_API_KEY") == "env-value"


def test_read_credential_falls_back_to_the_file(tmp_path, monkeypatch):
    cred_dir = tmp_path / "creds"
    cred_dir.mkdir()
    (cred_dir / "pod_id").write_text("pod-from-file\n")
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.setenv("WATCHDOG_CRED_DIR", str(cred_dir))

    assert _read_credential("pod_id", "RUNPOD_POD_ID") == "pod-from-file"


def test_main_refuses_to_run_with_no_credentials_anywhere(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.setenv("WATCHDOG_CRED_DIR", str(tmp_path / "missing"))

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
