"""ComfyUI's HTTP API - verified live 2026-10-17 against a real pod
(see docs/SOURCES.md "Wan 2.2 TI2V-5B real generation"): POST /prompt
returns {"prompt_id", "number", "node_errors"}; GET /history/<id> returns
{} while running and {<id>: {"status": {"status_str", "completed"},
"outputs": {...}}} once finished; GET /view?filename=&subfolder=&type=
streams the real output bytes. All three confirmed against a real job
that produced a playable 563KB mp4.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class ComfyUIError(Exception):
    pass


class ComfyUIClient(ABC):
    @abstractmethod
    def submit(self, prompt_body: dict) -> str:
        """POSTs a {"client_id", "prompt"} body (see wan_prompt_builder.py),
        returns the real prompt_id."""

    @abstractmethod
    def wait_for_completion(self, prompt_id: str, *, timeout_seconds: float, poll_interval_seconds: float) -> dict:
        """Blocks until /history/<prompt_id> reports completed, returns its
        "outputs" dict. Raises ComfyUIError on a failed run or on timeout."""

    @abstractmethod
    def download_output(self, filename: str, subfolder: str, type_: str) -> bytes:
        """Streams the real file bytes via /view."""


class RealComfyUIClient(ComfyUIClient):
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def submit(self, prompt_body: dict) -> str:
        import httpx

        response = httpx.post(f"{self._base_url}/prompt", json=prompt_body, timeout=30)
        response.raise_for_status()
        payload = response.json()
        node_errors = payload.get("node_errors") or {}
        if node_errors:
            raise ComfyUIError(f"ComfyUI rejected the prompt: {node_errors}")
        return payload["prompt_id"]

    def wait_for_completion(self, prompt_id: str, *, timeout_seconds: float = 1800, poll_interval_seconds: float = 5) -> dict:
        import httpx

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            response = httpx.get(f"{self._base_url}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            history = response.json()
            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("completed"):
                    if status.get("status_str") != "success":
                        raise ComfyUIError(f"ComfyUI run {prompt_id} finished without success: {status}")
                    return entry["outputs"]
                if status.get("status_str") == "error":
                    raise ComfyUIError(f"ComfyUI run {prompt_id} failed: {status}")
            time.sleep(poll_interval_seconds)
        raise ComfyUIError(f"ComfyUI run {prompt_id} did not complete within {timeout_seconds}s")

    def download_output(self, filename: str, subfolder: str, type_: str) -> bytes:
        import httpx

        response = httpx.get(
            f"{self._base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": type_},
            timeout=60,
        )
        response.raise_for_status()
        return response.content


class MockComfyUIClient(ComfyUIClient):
    """Scripted outputs, one per submit() call, in order - lets a test
    drive a whole episode's pipeline without a real GPU."""

    def __init__(self, scripted_outputs: list[dict] | None = None, scripted_bytes: bytes = b"fake-mp4-bytes"):
        self._scripted_outputs = list(scripted_outputs or [])
        self._scripted_bytes = scripted_bytes
        self.submitted_prompts: list[dict] = []
        self._next_id = 0

    def submit(self, prompt_body: dict) -> str:
        self.submitted_prompts.append(prompt_body)
        self._next_id += 1
        return f"mock-prompt-{self._next_id}"

    def wait_for_completion(self, prompt_id: str, *, timeout_seconds: float, poll_interval_seconds: float) -> dict:
        if not self._scripted_outputs:
            raise AssertionError("MockComfyUIClient ran out of scripted outputs")
        return self._scripted_outputs.pop(0)

    def download_output(self, filename: str, subfolder: str, type_: str) -> bytes:
        return self._scripted_bytes
