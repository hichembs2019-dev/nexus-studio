#!/usr/bin/env python3
"""Idle-shutdown safety net for the ComfyUI/Wan2.2 GPU pod.

Owner's explicit, hard requirement, 2026-10-22: a previous Runpod pod
kept billing by the hour while genuinely unused and cost real money,
because nothing ever stopped it. `RunpodGpuSession`
(src/integrations/runpod_client.py) is the PRIMARY mechanism - it starts
the pod only for the duration of one real production run and stops it
explicitly, success or failure, in under a minute of the job finishing.

This script is the independent SAFETY NET for the one gap that primary
mechanism cannot close on its own: the orchestrator process itself dying
before it reaches its own stop call (droplet crash, `kill -9`, a dropped
network partition after the job already finished, a bug that leaks the
`with RunpodGpuSession(...)` block). It runs ON THE POD ITSELF, so it
keeps working even if the droplet that started the job is gone.

Deliberately dependency-free (stdlib only: urllib, json) - it must run
inside whatever Python the official Runpod ComfyUI image ships, not a
curated environment this project controls.

Design notes:
- Runpod's own REST v2 API has NO field that stops a pod after a set
  time (verified live against Runpod's docs, 2026-10-22 - see
  runpod_client.py's module docstring for the exact source), and
  runpodctl's --stop-after/--terminate-after flags were never actually
  enforced server-side. The only real mechanism is a process, running
  somewhere, that calls the stop action itself - this script.
- Credentials come from local FILES
  (/workspace/.runpod_watchdog/{api_key,pod_id} by default), not
  inherited environment variables. Runpod's own documented gotcha: env
  vars set at pod creation reach the container's PID 1 only, never an
  SSH login shell - and this script is deployed and launched over SSH.
  Writing the real key/id to a file at deploy time sidesteps that
  entirely and is simple to verify by inspection.
- A queue read that fails outright (ComfyUI not yet up, crashed, or a
  transient error) counts the SAME as an idle queue, not as "unknown,
  assume busy" - a GPU with a broken ComfyUI service is exactly as
  wasteful as one with an empty queue, and the owner's real incident was
  exactly this kind of silent, undetected waste. Only a queue read that
  comes back with real work in it resets the idle timer.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

COMFYUI_QUEUE_URL = "http://127.0.0.1:8188/queue"
RUNPOD_API_BASE = "https://api.runpod.io/v2"

DEFAULT_CRED_DIR = "/workspace/.runpod_watchdog"

POLL_INTERVAL_S = int(os.environ.get("WATCHDOG_POLL_INTERVAL_S", "60"))
# 15 minutes of nothing real happening - generous enough that a normal
# gap between two queued jobs in the same production run never trips it,
# short enough that a real incident costs at most ~$0.15 on a $0.59/hr
# pod instead of hours.
IDLE_TIMEOUT_S = int(os.environ.get("WATCHDOG_IDLE_TIMEOUT_S", str(15 * 60)))
# ComfyUI itself can take a few minutes to finish booting (see the
# ComfyUI golden path's own ~4 min first-boot note) before the first
# real job is even submitted - the watchdog must not fire during that
# window just because nothing has started yet.
GRACE_PERIOD_S = int(os.environ.get("WATCHDOG_GRACE_PERIOD_S", str(10 * 60)))


class IdleTracker:
    """Pure decision logic, deliberately separated from real time/sleep
    so it can be unit-tested with synthetic timestamps."""

    def __init__(self, idle_timeout_s: float, grace_period_s: float, started_at: float):
        self._idle_timeout_s = idle_timeout_s
        self._grace_period_s = grace_period_s
        self._started_at = started_at
        self._last_busy_at = started_at

    def observe(self, idle: "bool | None", now: float) -> bool:
        """Feed one real poll result (True=queue empty, False=real work
        queued or running, None=queue unreachable). Returns True exactly
        when the pod should be stopped now."""
        if idle is False:
            self._last_busy_at = now
            return False
        if now - self._started_at < self._grace_period_s:
            return False
        return (now - self._last_busy_at) >= self._idle_timeout_s


def is_comfyui_idle() -> "bool | None":
    try:
        with urllib.request.urlopen(COMFYUI_QUEUE_URL, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    running = payload.get("queue_running", [])
    pending = payload.get("queue_pending", [])
    return not running and not pending


def _read_credential(filename: str, env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    cred_dir = os.environ.get("WATCHDOG_CRED_DIR", DEFAULT_CRED_DIR)
    try:
        with open(os.path.join(cred_dir, filename), "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def stop_pod(api_key: str, pod_id: str) -> None:
    req = urllib.request.Request(
        f"{RUNPOD_API_BASE}/pods/{pod_id}/action",
        data=json.dumps({"action": "stop"}).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        logging.info("idle watchdog: stop requested, Runpod responded %s", resp.status)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s idle-watchdog %(message)s")
    api_key = _read_credential("api_key", "RUNPOD_API_KEY")
    pod_id = _read_credential("pod_id", "RUNPOD_POD_ID")
    if not api_key or not pod_id:
        logging.error("idle watchdog: missing api_key or pod_id - refusing to run with no way to stop the pod")
        sys.exit(1)

    logging.info(
        "idle watchdog started: pod=%s idle_timeout=%ss poll_interval=%ss grace_period=%ss",
        pod_id, IDLE_TIMEOUT_S, POLL_INTERVAL_S, GRACE_PERIOD_S,
    )

    tracker = IdleTracker(IDLE_TIMEOUT_S, GRACE_PERIOD_S, time.monotonic())
    while True:
        time.sleep(POLL_INTERVAL_S)
        idle = is_comfyui_idle()
        if tracker.observe(idle, time.monotonic()):
            logging.warning("idle watchdog: idle timeout reached, stopping pod %s now", pod_id)
            try:
                stop_pod(api_key, pod_id)
            except Exception as exc:
                logging.error("idle watchdog: stop_pod failed (%s) - will retry on next poll", exc)
                continue
            return


if __name__ == "__main__":
    main()
