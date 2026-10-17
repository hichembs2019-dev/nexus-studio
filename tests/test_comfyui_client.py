from __future__ import annotations

import pytest

from src.integrations.comfyui_client import ComfyUIError, MockComfyUIClient, RealComfyUIClient


class _FakeResponse:
    def __init__(self, payload=None, content: bytes = b"", status_code: int = 200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_submit_posts_the_body_and_returns_the_real_prompt_id(monkeypatch):
    calls = []

    def fake_post(url, *, json, timeout):
        calls.append((url, json))
        return _FakeResponse({"prompt_id": "c63d9dbc-1dd8-4357-9082-7329717c16ca", "number": 0, "node_errors": {}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealComfyUIClient("https://pod-8188.proxy.runpod.net")
    prompt_id = client.submit({"client_id": "x", "prompt": {}})

    assert prompt_id == "c63d9dbc-1dd8-4357-9082-7329717c16ca"
    assert calls[0][0] == "https://pod-8188.proxy.runpod.net/prompt"


def test_submit_raises_on_node_errors(monkeypatch):
    def fake_post(url, *, json, timeout):
        return _FakeResponse({"prompt_id": "x", "node_errors": {"6": {"errors": ["missing input"]}}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealComfyUIClient("https://pod")
    with pytest.raises(ComfyUIError, match="missing input"):
        client.submit({"client_id": "x", "prompt": {}})


def test_wait_for_completion_polls_until_the_real_history_shows_success(monkeypatch):
    # Real shape verified live 2026-10-17: /history/<id> returns {} while
    # running, then {<id>: {"status": {...}, "outputs": {...}}} once done.
    responses = [_FakeResponse({}), _FakeResponse({}), _FakeResponse({
        "c63d9dbc": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"58": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}},
        }
    })]
    calls = []

    def fake_get(url, *, timeout=None, params=None):
        calls.append(url)
        return responses.pop(0)

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    client = RealComfyUIClient("https://pod")
    outputs = client.wait_for_completion("c63d9dbc", timeout_seconds=10, poll_interval_seconds=0)

    assert outputs == {"58": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}
    assert len(calls) == 3


def test_wait_for_completion_raises_on_a_failed_run(monkeypatch):
    def fake_get(url, *, timeout=None, params=None):
        return _FakeResponse({"pid": {"status": {"status_str": "error", "completed": True}, "outputs": {}}})

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealComfyUIClient("https://pod")
    with pytest.raises(ComfyUIError):
        client.wait_for_completion("pid", timeout_seconds=10, poll_interval_seconds=0)


def test_wait_for_completion_times_out_if_never_completed(monkeypatch):
    def fake_get(url, *, timeout=None, params=None):
        return _FakeResponse({})

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)
    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    def fake_sleep(seconds):
        fake_time[0] += seconds

    monkeypatch.setattr("time.sleep", fake_sleep)

    client = RealComfyUIClient("https://pod")
    with pytest.raises(ComfyUIError, match="did not complete"):
        client.wait_for_completion("pid", timeout_seconds=5, poll_interval_seconds=1)


def test_download_output_streams_real_bytes(monkeypatch):
    captured = {}

    def fake_get(url, *, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(content=b"\x00\x00\x00\x18ftypmp42fake-video-bytes")

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealComfyUIClient("https://pod")
    data = client.download_output("out.mp4", "", "output")

    assert data.startswith(b"\x00\x00\x00\x18ftyp")
    assert captured["params"] == {"filename": "out.mp4", "subfolder": "", "type": "output"}


def test_mock_comfyui_client_scripts_outputs_and_records_submissions():
    mock = MockComfyUIClient(scripted_outputs=[{"58": {"images": [{"filename": "a.mp4", "subfolder": "", "type": "output"}]}}])

    prompt_id = mock.submit({"prompt": {"1": {}}})
    outputs = mock.wait_for_completion(prompt_id, timeout_seconds=1, poll_interval_seconds=0)

    assert mock.submitted_prompts == [{"prompt": {"1": {}}}]
    assert outputs["58"]["images"][0]["filename"] == "a.mp4"
    assert mock.download_output("a.mp4", "", "output") == b"fake-mp4-bytes"
