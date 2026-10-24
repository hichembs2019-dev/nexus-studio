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

Automatic pod replacement (owner's instruction, 2026-10-23 - the real
reason this project stalled): a stopped pod's GPU is NOT reserved for
it, so `start` on a pinned pod id can fail with a real 400 "There are
not enough free GPUs on the host machine to start this pod." (seen live
twice - docs/SOURCES.md "RunPod host capacity" and "Real, confirmed
limit"), and the only fix that ever worked was a human creating a new
pod on the same network volume and pasting its new id into the config.
`ensure_pod_ready` now does that itself, in this order:
1. the pod last recorded as current (state file, see below), else the
   configured pod id;
2. any other existing pod mounted on the same network volume
   (`NETWORK_VOLUME_ID`) - a RUNNING one is reused as is, an EXITED one
   is started;
3. only if every one of those is missing or has no free GPU: a NEW pod,
   cloned field by field from a real existing pod's own GET /v2/pods/{id}
   config (image, disk, ports, env, cloud, data center, GPU type, network
   mount) - nothing here is guessed, so no pod spec is invented when no
   source pod exists at all (that raises instead).
Every v2 field used was read from Runpod's own published spec
(https://api.runpod.io/v2/openapi.json, fetched 2026-10-23):
CreatePodRequest {name, image, args, disk, ports, env, registry, cloud,
dataCenterIds, gpu {id, count, minCudaVersion}, mounts {network
[{volumeId, path}]}}, POST /v2/pods -> 201 + Pod, GET /v2/pods ->
{"pods": [...], "pagination": {"nextCursor", "hasNextPage"}}.
The replacement pod's id is written to a small JSON state file
(`RUNPOD_POD_STATE_PATH`, default `$OUTPUT_DIR/runpod_current_pod.json`)
so the next run goes straight to it instead of failing on the old id
again - the configured id itself (a GitHub secret, rewritten into .env
on every deploy) is never edited by this code.
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod

API_BASE = "https://api.runpod.io/v2"

# The one real network volume holding every Wan 2.2 / MultiTalk model
# file (docs/SOURCES.md, "The first real GPU pod" - 50GB, CA-MTL-1).
# Only pods mounted on it can run this project's graphs at all.
NETWORK_VOLUME_ID = "yamxrkrdos"

# Owner's hard rule (pod_idle_watchdog.py): never a GPU pod without
# automatic idle-shutdown. Only this custom image has the watchdog baked
# into its entrypoint (Dockerfile.comfyui), so a replacement is cloned
# only from a pod running it - never from one that could bill forever.
WATCHDOG_IMAGE_PREFIX = "ghcr.io/hichembs2019-dev/nexus-studio-comfyui"

# The custom image is built FROM runpod/comfyui:...-cuda13.0, and every
# real pod so far was created with gpu.minCudaVersion "13.0"
# (docs/SOURCES.md, "The first real GPU pod"). GET /v2/pods/{id} does not
# return that create-time constraint, so it is restated here.
MIN_CUDA_VERSION = "13.0"

# Real error texts, copied from the live responses recorded in
# docs/SOURCES.md - matched case-insensitively, as substrings.
_START_CAPACITY_MARKER = "not enough free gpus"
_CREATE_CAPACITY_MARKER = "no longer any instances available"

# Two straight creates hit "no longer any instances available" live and
# a later retry succeeded (docs/SOURCES.md, "Real, confirmed limit").
POD_CREATE_ATTEMPTS = 3
POD_CREATE_RETRY_INTERVAL_S = 30

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


class RunpodCapacityError(RunpodClientError):
    """No free GPU right now - for a `start`, on that pod's own host;
    for a create, anywhere matching the requested spec."""


class RunpodBalanceError(RunpodClientError):
    """Runpod answered 402 Payment Required: the account balance is too
    low to run a pod. No later line of the same run can succeed either."""


class RunpodPodNotFoundError(RunpodClientError):
    """GET /v2/pods/{id} returned 404 - the pod was terminated."""


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
        Raises RunpodPodNotFoundError on a 404, RunpodClientError on any
        other non-2xx response."""

    @abstractmethod
    def list_pods(self) -> list[dict]:
        """Returns every pod on the account (all pages). Raises
        RunpodClientError on a non-2xx response."""

    @abstractmethod
    def create_pod(self, body: dict) -> dict:
        """POST /v2/pods with a CreatePodRequest body, returns the new
        Pod. Raises RunpodCapacityError when Runpod reports no instance
        available, RunpodClientError on any other non-2xx response."""


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
            message = f"Runpod {action} on pod {pod_id} failed ({resp.status_code}): {resp.text[:500]}"
            if action == "start" and resp.status_code == 400 and _START_CAPACITY_MARKER in resp.text.lower():
                raise RunpodCapacityError(message)
            if resp.status_code == 402:
                raise RunpodBalanceError(message)
            raise RunpodClientError(message)
        try:
            return resp.json().get("status", "")
        except Exception:
            return ""

    def get_pod_status(self, pod_id: str) -> dict:
        import httpx

        resp = httpx.get(f"{API_BASE}/pods/{pod_id}", headers=self._headers(), timeout=30)
        if resp.status_code == 404:
            raise RunpodPodNotFoundError(f"Runpod pod {pod_id} not found (404): {resp.text[:500]}")
        if resp.status_code >= 400:
            raise RunpodClientError(f"Runpod status check for pod {pod_id} failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    def list_pods(self) -> list[dict]:
        import httpx

        pods: list[dict] = []
        params: dict = {}
        while True:
            resp = httpx.get(f"{API_BASE}/pods", headers=self._headers(), params=params, timeout=30)
            if resp.status_code >= 400:
                raise RunpodClientError(f"Runpod list pods failed ({resp.status_code}): {resp.text[:500]}")
            payload = resp.json()
            pods.extend(payload.get("pods", []))
            pagination = payload.get("pagination") or {}
            if not pagination.get("hasNextPage") or not pagination.get("nextCursor"):
                return pods
            params = {"cursor": pagination["nextCursor"]}

    def create_pod(self, body: dict) -> dict:
        import httpx

        resp = httpx.post(f"{API_BASE}/pods", headers=self._headers(), json=body, timeout=60)
        if resp.status_code >= 400:
            message = f"Runpod create pod failed ({resp.status_code}): {resp.text[:500]}"
            if resp.status_code == 400 and _CREATE_CAPACITY_MARKER in resp.text.lower():
                raise RunpodCapacityError(message)
            if resp.status_code == 402:
                raise RunpodBalanceError(message)
            raise RunpodClientError(message)
        return resp.json()


class MockRunpodClient(RunpodClient):
    """Scripted pod lifecycle, records every call - lets a test drive
    RunpodGpuSession with no real Runpod account and no real money.

    `pods` pre-seeds real-shaped Pod dicts by id (any other id starts as
    a bare EXITED pod, as before); `no_free_gpu` lists pod ids whose
    `start` fails with RunpodCapacityError; `missing` lists terminated
    ids (404); `create_capacity_failures` makes that many creates fail
    with RunpodCapacityError before one succeeds."""

    def __init__(
        self,
        *,
        comfyui_ready: bool = True,
        fail_stop: bool = False,
        pods: list[dict] | None = None,
        no_free_gpu: tuple[str, ...] = (),
        missing: tuple[str, ...] = (),
        create_capacity_failures: int = 0,
    ):
        self.calls: list[tuple[str, str]] = []
        self.created_bodies: list[dict] = []
        self._pods: dict[str, dict] = {p["id"]: dict(p) for p in (pods or [])}
        self._comfyui_ready = comfyui_ready
        self._fail_stop = fail_stop
        self._no_free_gpu = set(no_free_gpu)
        self._missing = set(missing)
        self._create_capacity_failures = create_capacity_failures

    def _pod(self, pod_id: str) -> dict:
        if pod_id in self._missing:
            raise RunpodPodNotFoundError(f"mock pod {pod_id} not found")
        return self._pods.setdefault(pod_id, {"id": pod_id, "status": "EXITED"})

    def start_pod(self, pod_id: str) -> str:
        self.calls.append(("start", pod_id))
        pod = self._pod(pod_id)
        if pod_id in self._no_free_gpu:
            raise RunpodCapacityError(f"mock: not enough free GPUs on the host machine to start {pod_id}")
        pod["status"] = "RUNNING"
        return pod["status"]

    def stop_pod(self, pod_id: str) -> str:
        self.calls.append(("stop", pod_id))
        if self._fail_stop:
            raise RunpodClientError("mock stop failure")
        pod = self._pod(pod_id)
        pod["status"] = "EXITED"
        return pod["status"]

    def get_pod_status(self, pod_id: str) -> dict:
        self.calls.append(("status", pod_id))
        return dict(self._pod(pod_id))

    def list_pods(self) -> list[dict]:
        self.calls.append(("list", ""))
        return [dict(p) for pid, p in self._pods.items() if pid not in self._missing]

    def create_pod(self, body: dict) -> dict:
        self.calls.append(("create", body.get("name", "")))
        self.created_bodies.append(body)
        if self._create_capacity_failures > 0:
            self._create_capacity_failures -= 1
            raise RunpodCapacityError("mock: there are no longer any instances available")
        new_id = f"mock-new-pod-{len(self.created_bodies)}"
        pod = {"id": new_id, "status": "RUNNING", **{k: v for k, v in body.items() if k != "dataCenterIds"}}
        self._pods[new_id] = pod
        return dict(pod)


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
        status = client.get_pod_status(pod_id).get("status")
        if status == "RUNNING":
            return
        # v2 `status` is the observed state - ERROR/TERMINATED never turn
        # into RUNNING, so waiting the full timeout on them only delays
        # the real error.
        if status in ("ERROR", "TERMINATED"):
            raise RunpodClientError(f"Pod {pod_id} went to {status} instead of RUNNING")
        time.sleep(POD_START_POLL_INTERVAL_S)
    raise RunpodClientError(f"Pod {pod_id} did not reach RUNNING within {POD_START_TIMEOUT_S}s")


def _default_state_path() -> str:
    return os.environ.get(
        "RUNPOD_POD_STATE_PATH",
        os.path.join(os.environ.get("OUTPUT_DIR", "./output"), "runpod_current_pod.json"),
    )


def _load_current_pod_id(state_path: str, configured_pod_id: str) -> str:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return configured_pod_id
    current = state.get(configured_pod_id) if isinstance(state, dict) else None
    return current if isinstance(current, str) and current else configured_pod_id


def _save_current_pod_id(state_path: str, configured_pod_id: str, current_pod_id: str) -> None:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    state[configured_pod_id] = current_pod_id
    try:
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError as exc:
        # Not fatal for this run (the new pod is already up), but the
        # next run would retry the old id first - said loudly, not hidden.
        _log.error("could not record replacement pod %s in %s (%s)", current_pod_id, state_path, exc)


def _on_volume(pod: dict, volume_id: str) -> bool:
    network = (pod.get("mounts") or {}).get("network") or []
    return any(m.get("volumeId") == volume_id for m in network)


def _replacement_body(source: dict, volume_id: str) -> dict:
    """Clones a real existing pod's own config into a CreatePodRequest.
    Refuses rather than guesses when the source is missing a field the
    new pod cannot run without."""
    image = source.get("image") or ""
    if not image.startswith(WATCHDOG_IMAGE_PREFIX):
        raise RunpodClientError(
            f"refusing to clone pod {source.get('id')}: image {image!r} is not the idle-watchdog image "
            f"{WATCHDOG_IMAGE_PREFIX} - a replacement without auto-stop could bill indefinitely"
        )
    network = [m for m in (source.get("mounts") or {}).get("network") or [] if m.get("volumeId") == volume_id]
    gpu = source.get("gpu") or {}
    if not network or not network[0].get("path") or not gpu.get("id"):
        raise RunpodClientError(f"refusing to clone pod {source.get('id')}: no network mount on {volume_id} or no GPU type")
    body: dict = {
        "name": source.get("name") or "nexus-studio-comfyui",
        "image": image,
        "gpu": {"id": gpu["id"], "count": gpu.get("count") or 1, "minCudaVersion": MIN_CUDA_VERSION},
        "mounts": {"network": [{"volumeId": volume_id, "path": network[0]["path"]}]},
    }
    for key in ("disk", "ports", "env", "cloud", "registry", "args"):
        if source.get(key):
            body[key] = source[key]
    # A network volume lives in one data center and a pod can only mount
    # it from there (NetworkMount: "in the same data center as the pod").
    if source.get("dataCenterId"):
        body["dataCenterIds"] = [source["dataCenterId"]]
    return body


def _create_replacement_pod(client: RunpodClient, source: dict, volume_id: str) -> str:
    body = _replacement_body(source, volume_id)
    last_error: Exception | None = None
    for attempt in range(1, POD_CREATE_ATTEMPTS + 1):
        try:
            pod = client.create_pod(body)
        except RunpodCapacityError as exc:
            last_error = exc
            _log.warning("create replacement pod: no capacity (attempt %d/%d): %s", attempt, POD_CREATE_ATTEMPTS, exc)
            if attempt < POD_CREATE_ATTEMPTS:
                time.sleep(POD_CREATE_RETRY_INTERVAL_S)
            continue
        new_id = pod.get("id")
        if not new_id:
            raise RunpodClientError(f"Runpod create pod returned no id: {pod!r}")
        _log.warning(
            "created replacement pod %s (cloned from %s, volume %s, gpu %s) - billing starts now",
            new_id, source.get("id"), volume_id, body["gpu"]["id"],
        )
        return new_id
    raise RunpodCapacityError(
        f"no {body['gpu']['id']} capacity in {body.get('dataCenterIds')} after {POD_CREATE_ATTEMPTS} attempts: {last_error}"
    )


def _start_or_replace(client: RunpodClient, pod_id: str, volume_id: str) -> str:
    """Returns the id of a pod that is RUNNING (or was just started /
    created) on `volume_id`, trying `pod_id` first."""
    source: dict | None = None
    try:
        pod = client.get_pod_status(pod_id)
    except RunpodPodNotFoundError:
        _log.warning("pod %s no longer exists - looking for another pod on volume %s", pod_id, volume_id)
    else:
        source = pod
        if pod.get("status") == "RUNNING":
            return pod_id
        try:
            client.start_pod(pod_id)
            return pod_id
        except RunpodCapacityError as exc:
            _log.warning("pod %s cannot start, no free GPU on its host: %s", pod_id, exc)

    others = [
        p for p in client.list_pods()
        if p.get("id") != pod_id and p.get("status") != "TERMINATED" and _on_volume(p, volume_id)
    ]
    for p in others:
        if p.get("status") == "RUNNING":
            return p["id"]
    for p in others:
        try:
            client.start_pod(p["id"])
            return p["id"]
        except RunpodCapacityError as exc:
            _log.warning("pod %s on the same volume cannot start either: %s", p["id"], exc)

    # GET /v2/pods leaves `runtime` null but returns the full config, so a
    # listed pod is as good a clone source as a fetched one.
    if source is None or not _on_volume(source, volume_id):
        source = next((p for p in others if (p.get("image") or "").startswith(WATCHDOG_IMAGE_PREFIX)), None)
    if source is None:
        raise RunpodClientError(
            f"pod {pod_id} is unusable and no existing pod on volume {volume_id} can be cloned - "
            "a new pod spec is never invented here; create one by hand once"
        )
    return _create_replacement_pod(client, source, volume_id)


def _resolve_and_start(
    client: RunpodClient, pod_id: str, *, state_path: str | None = None, volume_id: str = NETWORK_VOLUME_ID
) -> str:
    """Returns the id of the pod that is now running or starting (and
    billing) - kept separate from the readiness wait so a caller can
    know which pod to stop even if that wait then fails."""
    path = state_path or _default_state_path()
    current = _load_current_pod_id(path, pod_id)
    ready_id = _start_or_replace(client, current, volume_id)
    if ready_id != current:
        _save_current_pod_id(path, pod_id, ready_id)
    return ready_id


def _wait_until_ready(client: RunpodClient, pod_id: str) -> str:
    _wait_until_running(client, pod_id)
    _wait_until_comfyui_ready(pod_id)
    return f"https://{pod_id}-8188.proxy.runpod.net"


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


def ensure_pod_ready(
    client: RunpodClient, pod_id: str, *, state_path: str | None = None, volume_id: str = NETWORK_VOLUME_ID
) -> str:
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
    SSH".

    If `pod_id` cannot start for lack of a free GPU on its host (or no
    longer exists), another pod on the same network volume is used or a
    replacement is created - see this module's docstring. The returned
    URL is always the pod actually running, which may not be `pod_id`."""
    return _wait_until_ready(client, _resolve_and_start(client, pod_id, state_path=state_path, volume_id=volume_id))


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

    def __init__(self, client: RunpodClient, pod_id: str, *, state_path: str | None = None):
        self._client = client
        self._pod_id = pod_id
        self._state_path = state_path
        self.base_url: str | None = None

    def __enter__(self) -> "RunpodGpuSession":
        try:
            # The pod actually started may be a replacement - stop THAT
            # one on exit (or on a readiness failure just below), not the
            # configured id that never came up.
            self._pod_id = _resolve_and_start(self._client, self._pod_id, state_path=self._state_path)
            self.base_url = _wait_until_ready(self._client, self._pod_id)
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
