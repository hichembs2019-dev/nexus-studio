from __future__ import annotations

import json

import pytest

from src.integrations.remote_exec_client import MockRemoteExecClient, RealJupyterRemoteExecClient, RemoteExecError


class _FakeHttpxResponse:
    def __init__(self, json_payload=None):
        self._json = json_payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeHttpxClient:
    def __init__(self, cookies_after_login: dict):
        self.cookies = cookies_after_login
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return _FakeHttpxResponse()


def test_ensure_session_raises_without_an_xsrf_cookie(monkeypatch):
    import httpx

    fake_client = _FakeHttpxClient(cookies_after_login={"username-1": "abc"})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: fake_client)

    client = RealJupyterRemoteExecClient("https://pod-8888.proxy.runpod.net", "the-password")
    with pytest.raises(RemoteExecError, match="_xsrf"):
        client._ensure_session()


def test_ensure_session_populates_cookies_including_xsrf(monkeypatch):
    import httpx

    fake_client = _FakeHttpxClient(cookies_after_login={"_xsrf": "tok123", "username-1": "abc"})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: fake_client)

    client = RealJupyterRemoteExecClient("https://pod-8888.proxy.runpod.net", "the-password")
    client._ensure_session()

    assert client._cookies == {"_xsrf": "tok123", "username-1": "abc"}


def test_ensure_kernel_posts_with_the_xsrf_header_and_caches_the_kernel_id(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: _FakeHttpxClient({"_xsrf": "tok123"}))

    posts = []

    def fake_post(url, *, cookies, headers, json, timeout):
        posts.append((url, cookies, headers))
        return _FakeHttpxResponse({"id": "kernel-abc"})

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealJupyterRemoteExecClient("https://pod-8888.proxy.runpod.net", "the-password")
    client._ensure_kernel()

    assert client._kernel_id == "kernel-abc"
    assert posts[0][0] == "https://pod-8888.proxy.runpod.net/api/kernels"
    assert posts[0][2]["X-XSRFToken"] == "tok123"

    # Second call must not POST again - the kernel is cached.
    client._ensure_kernel()
    assert len(posts) == 1


class _FakeWebSocket:
    def __init__(self, messages: list[dict]):
        self._messages = [json.dumps(m) for m in messages]
        self.sent = []
        self.closed = False

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        return self._messages.pop(0)

    def close(self):
        self.closed = True


def _idle_status(msg_id: str, session_id: str) -> dict:
    return {"header": {"msg_type": "status"}, "parent_header": {"msg_id": msg_id}, "content": {"execution_state": "idle"}}


def test_run_sends_execute_request_and_collects_stream_output(monkeypatch):
    client = RealJupyterRemoteExecClient("https://pod-8888.proxy.runpod.net", "pw")
    client._cookies = {"_xsrf": "tok123"}
    client._kernel_id = "kernel-abc"

    import websocket

    # Build the fake socket after we know the msg_id the client generates by
    # intercepting uuid4, so the parent_header match works end to end.
    fixed_msg_id = "fixed-msg-id"
    fixed_session_id = "fixed-session-id"
    ids = iter([fixed_session_id, fixed_msg_id])
    monkeypatch.setattr("uuid.uuid4", lambda: type("U", (), {"hex": next(ids)})())

    captured = {}

    def fake_create_connection_final(url, *, header, timeout):
        captured["url"] = url
        captured["header"] = header
        return _FakeWebSocket(
            [
                {"header": {"msg_type": "stream"}, "parent_header": {"msg_id": fixed_msg_id}, "content": {"text": "hello "}},
                {"header": {"msg_type": "stream"}, "parent_header": {"msg_id": fixed_msg_id}, "content": {"text": "world"}},
                _idle_status(fixed_msg_id, fixed_session_id),
            ]
        )

    monkeypatch.setattr(websocket, "create_connection", fake_create_connection_final)

    output = client.run("print('hello world')", timeout_seconds=5)

    assert output == "hello world"
    assert captured["url"] == "wss://pod-8888.proxy.runpod.net/api/kernels/kernel-abc/channels?session_id=fixed-session-id"
    # The real pod returned 403 without these two headers on the websocket
    # upgrade itself, even with a valid session cookie - see docs/SOURCES.md.
    assert any(h.startswith("X-XSRFToken: tok123") for h in captured["header"])
    assert any(h.startswith("Origin: https://pod-8888.proxy.runpod.net") for h in captured["header"])


def test_run_raises_remote_exec_error_on_a_kernel_error_message(monkeypatch):
    client = RealJupyterRemoteExecClient("https://pod-8888.proxy.runpod.net", "pw")
    client._cookies = {"_xsrf": "tok123"}
    client._kernel_id = "kernel-abc"

    fixed_msg_id = "fixed-msg-id"
    ids = iter(["session-id", fixed_msg_id])

    import uuid as uuid_module

    monkeypatch.setattr(uuid_module, "uuid4", lambda: type("U", (), {"hex": next(ids)})())

    def fake_create_connection(url, *, header, timeout):
        return _FakeWebSocket(
            [{"header": {"msg_type": "error"}, "parent_header": {"msg_id": fixed_msg_id}, "content": {"traceback": ["Traceback...", "NameError: boom"]}}]
        )

    import websocket

    monkeypatch.setattr(websocket, "create_connection", fake_create_connection)

    with pytest.raises(RemoteExecError, match="NameError: boom"):
        client.run("undefined_name", timeout_seconds=5)


def test_mock_remote_exec_client_scripts_outputs_and_records_commands():
    mock = MockRemoteExecClient(scripted_outputs=["0\nok"])

    output = mock.run("subprocess.run(['ffmpeg', '-version'])")

    assert output == "0\nok"
    assert mock.commands == ["subprocess.run(['ffmpeg', '-version'])"]
