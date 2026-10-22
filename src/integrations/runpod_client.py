"""Runpod REST API v2 client for the ComfyUI/Wan2.2 GPU pod, plus
`RunpodGpuSession`, the orchestration piece that guarantees the pod is
stopped after every real production run.

Owner's explicit, hard requirement, 2026-10-22 ("يجب ان تضع له كود ايقاف
عند عدم الاستعمال، المشكلة السابقة لا نستعمله لكن هو يسحب من الرصيد لذلك
خسرت اموال فيه" - you must put in automatic stop code for when it's not
in use; the previous problem was we weren't using it but it kept
draining the balance, so I lost real money on it): a Runpod pod bills
continuously by the hour from the moment it is RUNNING until something
explicitly stops it - unlike Replicate, there is no per-call billing
here, so a pod left running after a crash, a dropped connection, or a
forgotten stop step bills in full for every idle hour.

Endpoint contract verified live against Runpod's own docs 2026-10-22
(via the connected Runpod MCP's search-docs/lookup-concept tools, not
guessed and not from training data):
- Base URL is https://api.runpod.io/v2 (NOT the deprecated v1
  rest.runpod.io host, which retires 2026-11-15).
- Every pod lifecycle change (start/stop/restart/terminate) is ONE
  endpoint: POST /v2/pods/{id}/action with body {"action": "start"} /
  {"action": "stop"} - v1's separate /start, /stop paths no longer
  exist in v2.
- GET /v2/pods/{id} returns the real-time status field: PROVISIONING,
  STARTING, RUNNING, EXITED, ERROR, TERMINATED.
- Auth is unchanged from v1: "Authorization: Bearer <RUNPOD_API_KEY>".
- Confirmed, documented fact (Runpod's own "Pod" concept page, rule
  pod.scheduled-stop): REST v2 has NO field that stops a pod after a
  set time, and runpodctl's own --stop-after/--terminate-after flags
  were never actually enforced server-side - this is exactly why this
  module exists instead of trusting either of those.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

API_BASE = "https://api.runpod.io/v2"

# Real-world pod boot time is dominated by container image pull / volume
# copy, not compute - the ComfyUI prebuilt image's own golden path
# documents ~4 minutes of proxy 502s on first boot. Generous margin kept
# for a cold data-center placement.
POD_START_TIMEOUT_S = 600
POD_START_POLL_INTERVAL_S = 10

# ComfyUI readiness is checked through the PUBLIC proxy (not a pod-
# internal address - this orchestration code runs on the droplet, not
# on the pod), the same /system_stats probe the ComfyUI golden path
# verifies with.
COMFYUI_READY_TIMEOUT_S = 600
COMFYUI_READY_POLL_INTERVAL_S = 10

_log = logging.getLogger(__name__)


class RunpodClientError(Exception):
    pass


class RunpodClient(ABC):
    @abstractmethod
    def start_pod(self, pod_id: str) -> str:
        """Requests the real pod transition to RUNNING. Returns whatever
        status the API reports back (not necessarily RUNNING yet - the
        transition is async). Raises RunpodClientError on a non-2xx
        response."""

    @abstractmethod
    def stop_pod(self, pod_id: str) -> str:
        """Requests the real pod transition to EXITED, which is what
        stops real hourly billing (compute cost reads 0.0 once EXITED -
        the volume disk, if any, keeps billing separately and is out of
        scope here). Raises RunpodClientError on a non-2xx response."""

    @abstractmethod
    def get_pod_status(self, pod_id: str) -> dict:
        """Returns the real, current pod resource as Runpod's API
        reports it right now (dict with at least a "status" key).
        Raises RunpodClientError on a non-2xx response."""


class RealRunpodClient(RunpodClient):
    def __init__(self, api_key: str):
        self._api_key = api_key

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def start_pod(self, pod_id: str) -> str:
        return self._action(pod_id, "start")

    def stop_pod(self, pod_id: str) -> str:
        return self._action(pod_id, "stop")

    def _action(self, pod_id: str, action: str) -> str:
        import httpx

        resp = httpx.post(
            f"{API_BASE}/pods/{pod_id}/action",
            headers=self._headers(),
            json={"action": action},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RunpodClientError(f"Runpod {action} on pod {pod_id} failed ({resp.status_code}): {resp.text[:500]}")
        try:
            return resp.json().get("status", "")
        except Exception:
            return ""

    def get_pod_status(self, pod_id: str) -> dict:
        import httpx

        resp = httpx.get(f"{API_BASE}/pods/{pod_id}", headers=self._headers(), timeout=30)
        if resp.status_code >= 400:
            raise RunpodClientError(f"Runpod status check for pod {pod_id} failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()


class MockRunpodClient(RunpodClient):
    """Scripted pod lifecycle, records every call - lets a test drive
    RunpodGpuSession with no real Runpod account and no real money."""

    def __init__(self, *, comfyui_ready: bool = True, fail_stop: bool = False):
        self.calls: list[tuple[str, str]] = []
        self._status = "EXITED"
        self._comfyui_ready = comfyui_ready
        self._fail_stop = fail_stop

    def start_pod(self, pod_id: str) -> str:
        self.calls.append(("start", pod_id))
        self._status = "RUNNING"
        return self._status

    def stop_pod(self, pod_id: str) -> str:
        self.calls.append(("stop", pod_id))
        if self._fail_stop:
            raise RunpodClientError("mock stop failure")
        self._status = "EXITED"
        return self._status

    def get_pod_status(self, pod_id: str) -> dict:
        self.calls.append(("status", pod_id))
        return {"status": self._status}


def deploy_idle_watchdog_via_ssh(
    *,
    ssh_host: str,
    ssh_port: int,
    ssh_key_path: str,
    api_key: str,
    pod_id: str,
    local_watchdog_path: str = "scripts/pod_idle_watchdog.py",
) -> None:
    """Installs and (re)launches scripts/pod_idle_watchdog.py on a
    running pod over SSH. Called once per RunpodGpuSession start, so a
    freshly-started pod always gets a fresh watchdog process - it is not
    required to survive a pod stop/start cycle on its own, because its
    whole job is to cover exactly the window between one real start and
    one real stop (see RunpodGpuSession's own docstring: the
    orchestrator's explicit stop is the primary mechanism; this is the
    safety net for that primary mechanism failing).

    Credentials are written to real FILES on the pod
    (/workspace/.runpod_watchdog/{api_key,pod_id}), never passed as
    environment variables - Runpod's own documented gotcha (the "pod"
    concept, rule pod.env-pid1-only, verified live 2026-09-28): env vars
    set at pod creation reach the container's PID 1 only, never an SSH
    login shell, so an env var written here would simply not exist for
    the watchdog process this same SSH session launches.

    UNVERIFIED against a real pod as of 2026-10-22 - no real Runpod pod
    exists yet to SSH into. Must be proven against a real pod (watchdog
    log inspected over SSH, then the pod driven idle and watched to
    actually stop) before this is reported as working - per this
    project's own standing rule, never claim something works without
    real evidence.
    """
    import subprocess

    ssh_base = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", ssh_key_path,
        "-p", str(ssh_port),
        f"root@{ssh_host}",
    ]

    subprocess.run(
        ssh_base + ["mkdir -p /workspace/.runpod_watchdog && pkill -f pod_idle_watchdog.py || true"],
        check=True, timeout=30,
    )
    with open(local_watchdog_path, "rb") as f:
        watchdog_source = f.read()
    subprocess.run(
        ssh_base + ["cat > /workspace/.runpod_watchdog/watchdog.py"],
        input=watchdog_source, check=True, timeout=30,
    )
    subprocess.run(
        ssh_base + ["cat > /workspace/.runpod_watchdog/api_key && chmod 600 /workspace/.runpod_watchdog/api_key"],
        input=api_key.encode(), check=True, timeout=30,
    )
    subprocess.run(
        ssh_base + ["cat > /workspace/.runpod_watchdog/pod_id"],
        input=pod_id.encode(), check=True, timeout=30,
    )
    subprocess.run(
        ssh_base + [
            "setsid python3 /workspace/.runpod_watchdog/watchdog.py "
            "> /workspace/.runpod_watchdog/watchdog.log 2>&1 < /dev/null &"
        ],
        check=True, timeout=30,
    )


def _wait_until_running(client: RunpodClient, pod_id: str) -> None:
    deadline = time.monotonic() + POD_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if client.get_pod_status(pod_id).get("status") == "RUNNING":
            return
        time.sleep(POD_START_POLL_INTERVAL_S)
    raise RunpodClientError(f"Pod {pod_id} did not reach RUNNING within {POD_START_TIMEOUT_S}s")


def _wait_until_comfyui_ready(pod_id: str) -> None:
    import httpx

    url = f"https://{pod_id}-8188.proxy.runpod.net/system_stats"
    deadline = time.monotonic() + COMFYUI_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=10)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(COMFYUI_READY_POLL_INTERVAL_S)
    raise RunpodClientError(f"ComfyUI on pod {pod_id} did not become ready within {COMFYUI_READY_TIMEOUT_S}s")


def ensure_pod_ready(client: RunpodClient, pod_id: str) -> str:
    """Starts the real pod if it is not already RUNNING, waits for it to
    reach RUNNING and for its ComfyUI to answer /system_stats, and
    returns the real public proxy base URL. Idempotent and cheap to call
    again on an already-running, already-ready pod (one status read plus
    one /system_stats probe - no real start/stop action).

    Unlike `RunpodGpuSession`, this never stops the pod - built for
    `SelfHostedS2VVideoClient`, which is called once per real dialogue
    line within one production run and must not pay the real ~minutes-
    long start+readiness cost on every single line. That caller relies
    entirely on the separately-proven on-pod idle watchdog
    (scripts/pod_idle_watchdog.py) to stop the real pod once the whole
    run has gone idle - see that script's own docstring and
    docs/SOURCES.md "The idle watchdog proven end to end, for real, zero
    SSH"."""
    status = client.get_pod_status(pod_id).get("status")
    if status != "RUNNING":
        client.start_pod(pod_id)
        _wait_until_running(client, pod_id)
    _wait_until_comfyui_ready(pod_id)
    return f"https://{pod_id}-8188.proxy.runpod.net"


class RunpodGpuSession:
    """Context manager wrapping one real production run on the GPU pod.

    This IS the primary idle-shutdown mechanism: the pod is started only
    for the duration of the `with` block and is explicitly stopped the
    moment it ends, success or failure - never left running "just in
    case". The independent, on-pod idle watchdog
    (scripts/pod_idle_watchdog.py) is the safety net for the one gap
    this class cannot close on its own: the stop call failing, or this
    process itself dying before reaching `finally` (droplet crash,
    `kill -9`, power loss) - see that script's own docstring.
    """

    def __init__(self, client: RunpodClient, pod_id: str):
        self._client = client
        self._pod_id = pod_id
        self.base_url: str | None = None

    def __enter__(self) -> "RunpodGpuSession":
        try:
            self.base_url = ensure_pod_ready(self._client, self._pod_id)
        except Exception:
            # __exit__ is never called when __enter__ itself raises
            # (Python context-manager protocol), so a readiness failure
            # here would otherwise leave a real pod running and billing
            # with nothing to stop it - the exact incident this whole
            # module exists to prevent.
            self._safe_stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._safe_stop()
        return False  # never swallow the real exception from the body

    def _safe_stop(self) -> bool:
        try:
            self._client.stop_pod(self._pod_id)
            return True
        except Exception as exc:
            _log.error(
                "RunpodGpuSession: stop_pod(%s) itself failed (%s) - the pod may "
                "still be running and billing. Relying on the on-pod idle watchdog "
                "to stop it instead.",
                self._pod_id,
                exc,
            )
            return False
