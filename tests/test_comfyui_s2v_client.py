from __future__ import annotations

import pytest

from src.integrations.comfyui_s2v_client import (
    ComfyUIS2VError,
    MockComfyUIS2VClient,
    RealComfyUIS2VClient,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict | None = None, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


def test_upload_file_posts_the_real_generic_upload_endpoint(monkeypatch):
    def fake_post(url, *, files=None, data=None, timeout=60):
        assert url == "https://pod.example/upload/image"
        assert files["image"] == ("line.mp3", b"real-audio-bytes", "audio/mpeg")
        assert data == {"type": "input"}
        return _FakeResponse(json_data={"name": "line.mp3", "subfolder": "", "type": "input"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealComfyUIS2VClient("https://pod.example")
    name = client.upload_file("line.mp3", b"real-audio-bytes", "audio/mpeg")

    assert name == "line.mp3"


def test_submit_posts_the_real_prompt_endpoint_and_returns_the_real_id(monkeypatch):
    def fake_post(url, *, json, timeout):
        assert url == "https://pod.example/prompt"
        return _FakeResponse(json_data={"prompt_id": "real-id", "number": 0, "node_errors": {}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealComfyUIS2VClient("https://pod.example")
    assert client.submit({"client_id": "x", "prompt": {}}) == "real-id"


def test_submit_raises_on_real_node_errors(monkeypatch):
    def fake_post(url, *, json, timeout):
        return _FakeResponse(json_data={"prompt_id": "x", "node_errors": {"39": {"errors": ["missing vae"]}}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealComfyUIS2VClient("https://pod.example")
    with pytest.raises(ComfyUIS2VError, match="missing vae"):
        client.submit({"client_id": "x", "prompt": {}})


def test_wait_for_completion_polls_until_success(monkeypatch):
    responses = iter([
        _FakeResponse(json_data={}),
        _FakeResponse(json_data={"real-id": {"status": {"status_str": "success", "completed": True}, "outputs": {"58": {"images": [{"filename": "out.mp4"}]}}}}),
    ])

    def fake_get(url, *, timeout):
        return next(responses)

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealComfyUIS2VClient("https://pod.example")
    outputs = client.wait_for_completion("real-id", timeout_seconds=30, poll_interval_seconds=1)

    assert outputs["58"]["images"][0]["filename"] == "out.mp4"


def test_wait_for_completion_raises_on_a_real_failed_run(monkeypatch):
    def fake_get(url, *, timeout):
        return _FakeResponse(json_data={"real-id": {"status": {"status_str": "error", "completed": False}}})

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = RealComfyUIS2VClient("https://pod.example")
    with pytest.raises(ComfyUIS2VError):
        client.wait_for_completion("real-id", timeout_seconds=30, poll_interval_seconds=1)


def test_download_output_streams_the_real_view_endpoint(monkeypatch):
    def fake_get(url, *, params, timeout):
        assert url == "https://pod.example/view"
        assert params == {"filename": "out.mp4", "subfolder": "", "type": "output"}
        return _FakeResponse(content=b"real-mp4-bytes")

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    client = RealComfyUIS2VClient("https://pod.example")
    assert client.download_output("out.mp4", "", "output") == b"real-mp4-bytes"


def test_mock_client_records_uploads_and_submissions_for_a_test_to_assert_on():
    client = MockComfyUIS2VClient(scripted_outputs=[{"58": {"images": [{"filename": "f.mp4", "subfolder": "", "type": "output"}]}}])

    name = client.upload_file("line.mp3", b"bytes", "audio/mpeg")
    prompt_id = client.submit({"client_id": "x", "prompt": {"1": {}}})
    outputs = client.wait_for_completion(prompt_id, timeout_seconds=1, poll_interval_seconds=1)

    assert name == "line.mp3"
    assert client.uploaded_files == [("line.mp3", b"bytes", "audio/mpeg")]
    assert client.submitted_prompts == [{"client_id": "x", "prompt": {"1": {}}}]
    assert outputs["58"]["images"][0]["filename"] == "f.mp4"
