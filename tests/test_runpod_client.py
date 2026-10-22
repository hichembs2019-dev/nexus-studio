from __future__ import annotations

import pytest

from src.integrations.runpod_client import (
    API_BASE,
    MockRunpodClient,
    RealRunpodClient,
    RunpodClientError,
    RunpodGpuSession,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


def test_start_pod_posts_the_real_v2_action_endpoint(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        assert url == f"{API_BASE}/pods/pod-123/action"
        assert headers["Authorization"] == "Bearer real-key"
        assert json == {"action": "start"}
        return _FakeResponse(json_data={"status": "STARTING"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealRunpodClient("real-key")
    assert client.start_pod("pod-123") == "STARTING"


def test_stop_pod_posts_the_real_v2_action_endpoint_with_the_stop_action(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        assert json == {"action": "stop"}
        return _FakeResponse(json_data={"status": "EXITED"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealRunpodClient("real-key")
    assert client.stop_pod("pod-123") == "EXITED"


def test_get_pod_status_reads_the_real_pod_resource(monkeypatch):
    def fake_get(url, *, headers, timeout):
        assert url == f"{API_BASE}/pods/pod-123"
        return _FakeResponse(json_data={"status": "RUNNING", "id": "pod-123"})

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealRunpodClient("real-key")
    assert client.get_pod_status("pod-123") == {"status": "RUNNING", "id": "pod-123"}


def test_a_failed_action_raises_runpodclienterror(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(status_code=401, text="unauthorized"))

    client = RealRunpodClient("bad-key")
    with pytest.raises(RunpodClientError):
        client.start_pod("pod-123")


def test_a_failed_status_read_raises_runpodclienterror(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(status_code=404, text="not found"))

    client = RealRunpodClient("real-key")
    with pytest.raises(RunpodClientError):
        client.get_pod_status("pod-123")


def _ready_now(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(status_code=200))


def test_session_starts_the_pod_only_when_it_is_not_already_running(monkeypatch):
    _ready_now(monkeypatch)
    client = MockRunpodClient()

    with RunpodGpuSession(client, "pod-123"):
        pass

    assert ("start", "pod-123") in client.calls
    assert ("stop", "pod-123") in client.calls


def test_session_skips_start_when_the_pod_is_already_running(monkeypatch):
    _ready_now(monkeypatch)
    client = MockRunpodClient()
    client.start_pod("pod-123")  # pre-seed the pod as already RUNNING
    client.calls.clear()

    with RunpodGpuSession(client, "pod-123"):
        pass

    assert ("start", "pod-123") not in client.calls
    assert ("stop", "pod-123") in client.calls


def test_session_stops_the_pod_even_when_the_real_work_raises(monkeypatch):
    _ready_now(monkeypatch)
    client = MockRunpodClient()

    with pytest.raises(ValueError, match="real production failure"):
        with RunpodGpuSession(client, "pod-123"):
            raise ValueError("real production failure")

    assert ("stop", "pod-123") in client.calls


def test_session_stops_the_pod_even_when_comfyui_never_becomes_ready(monkeypatch):
    import httpx

    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("time.sleep", lambda _s: clock.__setitem__("t", clock["t"] + 10_000))
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(status_code=502))

    client = MockRunpodClient()

    with pytest.raises(RunpodClientError):
        with RunpodGpuSession(client, "pod-123"):
            raise AssertionError("the body must never run - readiness never succeeded")

    assert ("stop", "pod-123") in client.calls


def test_a_failed_stop_call_does_not_mask_the_real_bodys_exception(monkeypatch):
    _ready_now(monkeypatch)
    client = MockRunpodClient(fail_stop=True)

    with pytest.raises(ValueError, match="real production failure"):
        with RunpodGpuSession(client, "pod-123"):
            raise ValueError("real production failure")


def test_a_failed_stop_call_after_a_successful_run_does_not_itself_raise(monkeypatch):
    # Mutation check: a version that lets the stop failure propagate
    # would turn "pod successfully produced the video" into a reported
    # crash - the owner never asked for that, and __exit__ swallowing
    # (and loudly logging) the stop failure is the one designed exception
    # to "never silently swallow a real error" (the idle watchdog is the
    # documented, deliberate safety net for exactly this case).
    _ready_now(monkeypatch)
    client = MockRunpodClient(fail_stop=True)

    with RunpodGpuSession(client, "pod-123"):
        pass

    assert ("stop", "pod-123") in client.calls
