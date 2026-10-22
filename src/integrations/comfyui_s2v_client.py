"""ComfyUI's HTTP API for the self-hosted Wan 2.2 S2V pipeline - verified
live 2026-10-22 against a real pod (`wt5ennaph7w5nh`, custom image with
the idle watchdog baked in): a real uploaded photo + a real uploaded
speech clip produced a real, playable 695,215-byte mp4 (640x640 h264,
aac audio, 5.3125s) - see docs/SOURCES.md "Self-hosted Wan 2.2 S2V: a
real cinematic clip, proven end to end".

Contract (identical in shape to the proven TI2V-5B client,
comfyui_client.py, which this intentionally mirrors rather than
diverges from):
- POST /prompt returns {"prompt_id", "number", "node_errors"}.
- GET /history/<id> returns {} while running and {<id>: {"status":
  {"status_str", "completed"}, "outputs": {...}}} once finished.
- GET /view?filename=&subfolder=&type= streams the real output bytes.
- POST /upload/image (the name is generic - confirmed live to accept
  any file type, not just images) with a multipart field named "image"
  plus "type": "input" uploads a real file into ComfyUI's input
  directory and returns {"name", "subfolder", "type"} - "name" is what
  LoadImage/LoadAudio reference by filename.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class ComfyUIS2VError(Exception):
    pass


class ComfyUIS2VClient(ABC):
    @abstractmethod
    def upload_file(self, filename: str, content: bytes, content_type: str) -> str:
        """Uploads a real file (reference photo or audio clip) to the
        pod's ComfyUI input directory. Returns the real filename to
        reference from a LoadImage/LoadAudio node - not necessarily the
        same as the input `filename` (ComfyUI may rename on collision)."""

    @abstractmethod
    def submit(self, prompt_body: dict) -> str:
        """POSTs a {"client_id", "prompt"} body (see
        wan_s2v_prompt_builder.py), returns the real prompt_id. Raises
        ComfyUIS2VError if ComfyUI rejects the graph (node_errors)."""

    @abstractmethod
    def wait_for_completion(self, prompt_id: str, *, timeout_seconds: float, poll_interval_seconds: float) -> dict:
        """Blocks until /history/<prompt_id> reports completed, returns
        its "outputs" dict. Raises ComfyUIS2VError on a failed run or on
        timeout."""

    @abstractmethod
    def download_output(self, filename: str, subfolder: str, type_: str) -> bytes:
        """Streams the real file bytes via /view."""


class RealComfyUIS2VClient(ComfyUIS2VClient):
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def upload_file(self, filename: str, content: bytes, content_type: str) -> str:
        import httpx

        response = httpx.post(
            f"{self._base_url}/upload/image",
            files={"image": (filename, content, content_type)},
            data={"type": "input"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["name"]

    def submit(self, prompt_body: dict) -> str:
        import httpx

        response = httpx.post(f"{self._base_url}/prompt", json=prompt_body, timeout=30)
        response.raise_for_status()
        payload = response.json()
        node_errors = payload.get("node_errors") or {}
        if node_errors:
            raise ComfyUIS2VError(f"ComfyUI rejected the S2V prompt: {node_errors}")
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
                        raise ComfyUIS2VError(f"ComfyUI S2V run {prompt_id} finished without success: {status}")
                    return entry["outputs"]
                if status.get("status_str") == "error":
                    raise ComfyUIS2VError(f"ComfyUI S2V run {prompt_id} failed: {status}")
            time.sleep(poll_interval_seconds)
        raise ComfyUIS2VError(f"ComfyUI S2V run {prompt_id} did not complete within {timeout_seconds}s")

    def download_output(self, filename: str, subfolder: str, type_: str) -> bytes:
        import httpx

        response = httpx.get(
            f"{self._base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": type_},
            timeout=60,
        )
        response.raise_for_status()
        return response.content


class MockComfyUIS2VClient(ComfyUIS2VClient):
    """Scripted outputs, one per submit() call, in order - lets a test
    drive a whole S2V production without a real GPU."""

    def __init__(self, scripted_outputs: list[dict] | None = None, scripted_bytes: bytes = b"fake-s2v-mp4-bytes"):
        self._scripted_outputs = list(scripted_outputs or [])
        self._scripted_bytes = scripted_bytes
        self.uploaded_files: list[tuple[str, bytes, str]] = []
        self.submitted_prompts: list[dict] = []
        self._next_id = 0
        self._next_upload_name = 0

    def upload_file(self, filename: str, content: bytes, content_type: str) -> str:
        self.uploaded_files.append((filename, content, content_type))
        self._next_upload_name += 1
        return filename

    def submit(self, prompt_body: dict) -> str:
        self.submitted_prompts.append(prompt_body)
        self._next_id += 1
        return f"mock-s2v-prompt-{self._next_id}"

    def wait_for_completion(self, prompt_id: str, *, timeout_seconds: float, poll_interval_seconds: float) -> dict:
        if not self._scripted_outputs:
            raise AssertionError("MockComfyUIS2VClient ran out of scripted outputs")
        return self._scripted_outputs.pop(0)

    def download_output(self, filename: str, subfolder: str, type_: str) -> bytes:
        return self._scripted_bytes
