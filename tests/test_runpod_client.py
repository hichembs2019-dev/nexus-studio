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


# --- automatic pod replacement (no free GPU on the pinned pod's host) ---

from src.integrations.runpod_client import (  # noqa: E402
    MIN_CUDA_VERSION,
    NETWORK_VOLUME_ID,
    RunpodCapacityError,
    RunpodPodNotFoundError,
    ensure_pod_ready,
)

_IMAGE = "ghcr.io/hichembs2019-dev/nexus-studio-comfyui:latest"


def _real_pod(pod_id: str, *, status: str = "EXITED", volume: str = NETWORK_VOLUME_ID, image: str = _IMAGE) -> dict:
    # Same shape as a real v2 Pod (openapi.json `Pod` schema).
    return {
        "id": pod_id,
        "name": "nexus-studio-comfyui",
        "status": status,
        "image": image,
        "args": "",
        "disk": 50,
        "ports": ["8188/http", "8888/http", "22/tcp"],
        "env": {"JUPYTER_PASSWORD": "pw"},
        "registry": None,
        "cloud": "SECURE",
        "dataCenterId": "CA-MTL-1",
        "gpu": {"id": "NVIDIA A40", "count": 1, "vcpuCount": 9, "memory": 50},
        "mounts": {"network": [{"volumeId": volume, "path": "/workspace"}]},
    }


def test_start_on_a_host_with_no_free_gpu_raises_the_capacity_error(monkeypatch):
    import httpx

    text = '{"title":"Bad Request","status":400,"detail":"There are not enough free GPUs on the host machine to start this pod."}'
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(status_code=400, text=text))
    with pytest.raises(RunpodCapacityError):
        RealRunpodClient("real-key").start_pod("pod-123")


def test_an_unrelated_400_on_start_is_not_treated_as_capacity(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(status_code=400, text="request could not be processed"))
    with pytest.raises(RunpodClientError) as info:
        RealRunpodClient("real-key").start_pod("pod-123")
    assert not isinstance(info.value, RunpodCapacityError)


def test_get_pod_status_404_raises_not_found(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(status_code=404, text="pod not found"))
    with pytest.raises(RunpodPodNotFoundError):
        RealRunpodClient("real-key").get_pod_status("pod-123")


def test_list_pods_reads_the_v2_envelope_and_follows_every_page(monkeypatch):
    import httpx

    seen = []

    def fake_get(url, *, headers, params, timeout):
        assert url == f"{API_BASE}/pods"
        seen.append(dict(params))
        if not params:
            return _FakeResponse(json_data={"pods": [{"id": "a"}], "pagination": {"nextCursor": "c1", "hasNextPage": True}})
        return _FakeResponse(json_data={"pods": [{"id": "b"}], "pagination": {"nextCursor": None, "hasNextPage": False}})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert [p["id"] for p in RealRunpodClient("real-key").list_pods()] == ["a", "b"]
    assert seen == [{}, {"cursor": "c1"}]


def test_create_pod_posts_the_v2_body_and_flags_no_instances_as_capacity(monkeypatch):
    import httpx

    def fake_post(url, *, headers, json, timeout):
        assert url == f"{API_BASE}/pods"
        assert json == {"name": "x"}
        return _FakeResponse(status_code=201, json_data={"id": "new-1"})

    monkeypatch.setattr(httpx, "post", fake_post)
    assert RealRunpodClient("real-key").create_pod({"name": "x"}) == {"id": "new-1"}

    text = '{"detail": "There are no longer any instances available with the requested specifications. Please refresh and try again."}'
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(status_code=400, text=text))
    with pytest.raises(RunpodCapacityError):
        RealRunpodClient("real-key").create_pod({"name": "x"})


def test_no_free_gpu_reuses_another_running_pod_on_the_same_volume(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    client = MockRunpodClient(
        pods=[_real_pod("old"), _real_pod("other", status="RUNNING")], no_free_gpu=("old",)
    )

    url = ensure_pod_ready(client, "old", state_path=str(tmp_path / "s.json"))

    assert url == "https://other-8188.proxy.runpod.net"
    assert client.created_bodies == []


def test_no_free_gpu_anywhere_creates_a_clone_on_the_same_volume(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    client = MockRunpodClient(pods=[_real_pod("old")], no_free_gpu=("old",))

    url = ensure_pod_ready(client, "old", state_path=str(tmp_path / "s.json"))

    assert url == "https://mock-new-pod-1-8188.proxy.runpod.net"
    assert client.created_bodies == [{
        "name": "nexus-studio-comfyui",
        "image": _IMAGE,
        "gpu": {"id": "NVIDIA A40", "count": 1, "minCudaVersion": MIN_CUDA_VERSION},
        "mounts": {"network": [{"volumeId": NETWORK_VOLUME_ID, "path": "/workspace"}]},
        "disk": 50,
        "ports": ["8188/http", "8888/http", "22/tcp"],
        "env": {"JUPYTER_PASSWORD": "pw"},
        "cloud": "SECURE",
        "dataCenterIds": ["CA-MTL-1"],
    }]


def test_the_replacement_is_remembered_so_the_next_run_skips_the_dead_pod(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    state = str(tmp_path / "s.json")
    client = MockRunpodClient(pods=[_real_pod("old")], no_free_gpu=("old",))
    ensure_pod_ready(client, "old", state_path=state)
    client.calls.clear()

    url = ensure_pod_ready(client, "old", state_path=state)

    assert url == "https://mock-new-pod-1-8188.proxy.runpod.net"
    assert ("start", "old") not in client.calls
    assert len(client.created_bodies) == 1


def test_a_terminated_configured_pod_is_replaced_from_a_listed_pod_on_the_volume(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    client = MockRunpodClient(
        pods=[_real_pod("gone"), _real_pod("sibling")], missing=("gone",), no_free_gpu=("sibling",)
    )

    url = ensure_pod_ready(client, "gone", state_path=str(tmp_path / "s.json"))

    assert url == "https://mock-new-pod-1-8188.proxy.runpod.net"
    assert client.created_bodies[0]["mounts"]["network"][0]["volumeId"] == NETWORK_VOLUME_ID


def test_pods_on_another_volume_are_never_used_or_cloned(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    client = MockRunpodClient(
        pods=[_real_pod("foreign", status="RUNNING", volume="othervol")], missing=("old",)
    )

    with pytest.raises(RunpodClientError, match="no existing pod on volume"):
        ensure_pod_ready(client, "old", state_path=str(tmp_path / "s.json"))
    assert client.created_bodies == []


def test_never_clones_a_pod_without_the_idle_watchdog_image(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    client = MockRunpodClient(
        pods=[_real_pod("old", image="runpod/comfyui:1.3.3-comfyuiv0.30.0-cuda13.0")], no_free_gpu=("old",)
    )

    with pytest.raises(RunpodClientError, match="idle-watchdog image"):
        ensure_pod_ready(client, "old", state_path=str(tmp_path / "s.json"))
    assert client.created_bodies == []


def test_create_is_retried_on_no_capacity_then_gives_up(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = MockRunpodClient(pods=[_real_pod("old")], no_free_gpu=("old",), create_capacity_failures=2)
    assert ensure_pod_ready(client, "old", state_path=str(tmp_path / "a.json")) == "https://mock-new-pod-3-8188.proxy.runpod.net"

    client = MockRunpodClient(pods=[_real_pod("old")], no_free_gpu=("old",), create_capacity_failures=99)
    with pytest.raises(RunpodCapacityError):
        ensure_pod_ready(client, "old", state_path=str(tmp_path / "b.json"))
    assert len(client.created_bodies) == 3


def test_session_stops_the_replacement_pod_not_the_dead_configured_one(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    # Fake clock: a session still tracking the dead id would poll it for
    # RUNNING forever - this makes that fail in one step, not 600s.
    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("time.sleep", lambda _s: clock.__setitem__("t", clock["t"] + 10_000))
    client = MockRunpodClient(pods=[_real_pod("old")], no_free_gpu=("old",))

    with RunpodGpuSession(client, "old", state_path=str(tmp_path / "s.json")) as session:
        assert session.base_url == "https://mock-new-pod-1-8188.proxy.runpod.net"

    assert ("stop", "mock-new-pod-1") in client.calls


def test_session_stops_the_replacement_pod_when_its_comfyui_never_becomes_ready(monkeypatch, tmp_path):
    import httpx

    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("time.sleep", lambda _s: clock.__setitem__("t", clock["t"] + 10_000))
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(status_code=502))
    client = MockRunpodClient(pods=[_real_pod("old")], no_free_gpu=("old",))

    with pytest.raises(RunpodClientError):
        with RunpodGpuSession(client, "old", state_path=str(tmp_path / "s.json")):
            raise AssertionError("the body must never run")

    assert ("stop", "mock-new-pod-1") in client.calls


def test_waiting_for_running_fails_fast_on_error_status(monkeypatch, tmp_path):
    _ready_now(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _s: (_ for _ in ()).throw(AssertionError("must not wait")))
    client = MockRunpodClient(pods=[_real_pod("old", status="ERROR")])
    client.start_pod = lambda pod_id: "ERROR"  # start accepted, pod stays ERROR

    with pytest.raises(RunpodClientError, match="went to ERROR"):
        ensure_pod_ready(client, "old", state_path=str(tmp_path / "s.json"))


def test_a_402_low_balance_on_start_is_a_balance_error(monkeypatch):
    import httpx

    from src.integrations.runpod_client import RunpodBalanceError

    body = '{"detail":"Your account balance is too low to rent a pod. Please add funds to your account.","status":402}'
    monkeypatch.setattr(httpx, "post", lambda url, *, headers, json, timeout: _FakeResponse(402, text=body))

    with pytest.raises(RunpodBalanceError, match="balance is too low"):
        RealRunpodClient("real-key").start_pod("pod-123")
