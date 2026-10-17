"""Runs real shell commands on the RunPod pod without SSH.

Direct SSH (port 22) is blocked by this project's own execution sandbox
(verified live 2026-10-17: a real connection to ssh.runpod.io:22 timed out,
and the sandbox's own proxy status confirmed only HTTP/HTTPS is proxied -
see docs/SOURCES.md "SSH blocked, Jupyter remote-exec works"). The pod's
Jupyter server (port 8888/http) goes through the same HTTPS proxy and was
proven live the same day: token login, kernel start, and a real
`nvidia-smi` call returning the real GPU name over the kernel's websocket
channel.

Two cookie/auth details that are easy to get wrong and were both found by
testing, not by reading docs:
- the cookie jar's HttpOnly line is prefixed "#HttpOnly_" - skipping every
  "#"-prefixed line (the naive reading of the Netscape format) silently
  drops the session identity cookie and the websocket handshake 403s with
  no useful message.
- the websocket upgrade itself 403s without an explicit Origin header
  matching the Jupyter host, even with a valid session cookie and a valid
  X-XSRFToken.
"""
from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod


class RemoteExecError(Exception):
    pass


class RemoteExecClient(ABC):
    @abstractmethod
    def run(self, code: str, *, timeout_seconds: float = 120) -> str:
        """Executes Python code on the remote kernel, returns its stdout
        (stream output). Use subprocess from within `code` to run real
        shell commands (ffmpeg, find, du, ...). Raises RemoteExecError on
        a Python-level error in the remote code."""


class RealJupyterRemoteExecClient(RemoteExecClient):
    def __init__(self, base_url: str, password: str):
        self._base_url = base_url.rstrip("/")
        self._password = password
        self._cookies: dict[str, str] = {}
        self._kernel_id: str | None = None

    def _ensure_session(self) -> None:
        import httpx

        if self._cookies:
            return
        client = httpx.Client(follow_redirects=True)
        login = client.get(f"{self._base_url}/login", params={"token": self._password}, timeout=20)
        login.raise_for_status()
        tree = client.get(f"{self._base_url}/tree", timeout=20)
        for name, value in client.cookies.items():
            self._cookies[name] = value
        if "_xsrf" not in self._cookies:
            raise RemoteExecError("Jupyter login did not yield an _xsrf cookie - check the password/token")

    def _ensure_kernel(self) -> None:
        import httpx

        self._ensure_session()
        if self._kernel_id:
            return
        response = httpx.post(
            f"{self._base_url}/api/kernels",
            cookies=self._cookies,
            headers={"X-XSRFToken": self._cookies["_xsrf"], "Content-Type": "application/json"},
            json={},
            timeout=30,
        )
        response.raise_for_status()
        self._kernel_id = response.json()["id"]

    def run(self, code: str, *, timeout_seconds: float = 120) -> str:
        import websocket

        self._ensure_kernel()
        host = self._base_url.split("://", 1)[1]
        scheme = "wss" if self._base_url.startswith("https") else "ws"
        session_id = uuid.uuid4().hex
        url = f"{scheme}://{host}/api/kernels/{self._kernel_id}/channels?session_id={session_id}"
        cookie_header = "; ".join(f"{k}={v}" for k, v in self._cookies.items())

        ws = websocket.create_connection(
            url,
            header=[
                f"Cookie: {cookie_header}",
                f"X-XSRFToken: {self._cookies['_xsrf']}",
                f"Origin: {self._base_url}",
            ],
            timeout=timeout_seconds,
        )
        try:
            msg_id = uuid.uuid4().hex
            ws.send(json.dumps(self._execute_request(code, msg_id, session_id)))
            return self._collect_output(ws, msg_id, timeout_seconds)
        finally:
            ws.close()

    @staticmethod
    def _execute_request(code: str, msg_id: str, session_id: str) -> dict:
        return {
            "header": {
                "msg_id": msg_id,
                "username": "nexus-studio",
                "session": session_id,
                "msg_type": "execute_request",
                "version": "5.3",
                "date": "",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
            "channel": "shell",
        }

    @staticmethod
    def _collect_output(ws, msg_id: str, timeout_seconds: float) -> str:
        outputs: list[str] = []
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            raw = ws.recv()
            if not raw:
                continue
            message = json.loads(raw)
            if message.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = message.get("header", {}).get("msg_type")
            content = message.get("content", {})
            if msg_type == "stream":
                outputs.append(content.get("text", ""))
            elif msg_type == "error":
                raise RemoteExecError("\n".join(content.get("traceback", [])))
            elif msg_type == "status" and content.get("execution_state") == "idle":
                return "".join(outputs)
        raise RemoteExecError(f"Remote execution did not reach idle state within {timeout_seconds}s")


class MockRemoteExecClient(RemoteExecClient):
    """Returns scripted stdout, one per run() call, in order."""

    def __init__(self, scripted_outputs: list[str] | None = None):
        self._outputs = list(scripted_outputs or [])
        self.commands: list[str] = []

    def run(self, code: str, *, timeout_seconds: float = 120) -> str:
        self.commands.append(code)
        if not self._outputs:
            raise AssertionError("MockRemoteExecClient ran out of scripted outputs")
        return self._outputs.pop(0)
