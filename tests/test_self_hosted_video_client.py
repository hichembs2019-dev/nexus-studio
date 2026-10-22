from __future__ import annotations

import base64
import subprocess

import pytest

from src.integrations.comfyui_s2v_client import MockComfyUIS2VClient
from src.integrations.replicate_video_client import ReplicateVideoError
from src.integrations.runpod_client import MockRunpodClient
from src.integrations.self_hosted_video_client import (
    DEFAULT_HOURLY_RATE_USD,
    SelfHostedS2VError,
    SelfHostedS2VVideoClient,
    _decode_data_uri,
    self_hosted_price_per_second_usd,
)
from src.workflows.wan_s2v_prompt_builder import VAE_NAME


def _data_uri(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _fake_ffprobe(monkeypatch, duration: str = "5.232000"):
    def fake_run(cmd, *, capture_output, text):
        assert cmd[0] == "ffprobe"
        return subprocess.CompletedProcess(cmd, 0, stdout=duration, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_decode_data_uri_round_trips_real_bytes():
    raw, mime = _decode_data_uri(_data_uri(b"real-bytes", "image/jpeg"))
    assert raw == b"real-bytes"
    assert mime == "image/jpeg"


def test_decode_data_uri_rejects_a_non_data_uri():
    with pytest.raises(SelfHostedS2VError):
        _decode_data_uri("https://example.com/not-a-data-uri.png")


def test_self_hosted_price_per_second_usd_divides_the_real_hourly_rate_not_a_guess():
    # Mutation check: a version using *60 (per-minute) instead of *3600
    # (per-second) would still look plausible - pin the real arithmetic.
    assert self_hosted_price_per_second_usd(3600.0) == 1.0
    assert self_hosted_price_per_second_usd(DEFAULT_HOURLY_RATE_USD) == pytest.approx(DEFAULT_HOURLY_RATE_USD / 3600.0)


def test_generate_speech_to_video_ensures_the_pod_is_ready_uploads_builds_and_downloads(monkeypatch):
    _fake_ffprobe(monkeypatch)
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"58": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}],
        scripted_bytes=b"real-mp4-bytes",
    )

    client = SelfHostedS2VVideoClient(runpod, "wt5ennaph7w5nh", comfyui_client_factory=lambda base_url: comfy)

    result = client.generate_speech_to_video(
        prompt="a woman speaking warmly",
        image_data_uri=_data_uri(b"photo-bytes", "image/jpeg"),
        audio_data_uri=_data_uri(b"audio-bytes", "audio/mpeg"),
    )

    assert result == b"real-mp4-bytes"
    assert ("start", "wt5ennaph7w5nh") in runpod.calls  # pod was not already running
    assert comfy.uploaded_files[0] == ("reference.jpg", b"photo-bytes", "image/jpeg")
    assert comfy.uploaded_files[1] == ("line.mp3", b"audio-bytes", "audio/mpeg")
    submitted = comfy.submitted_prompts[0]["prompt"]
    assert submitted["61"]["inputs"]["image"] == "reference.jpg"
    assert submitted["62"]["inputs"]["audio"] == "line.mp3"
    assert submitted["39"]["inputs"]["vae_name"] == VAE_NAME
    # Real audio duration (5.232s, from the faked ffprobe) must drive the
    # real frame count, not a hard-coded default.
    assert submitted["55"]["inputs"]["length"] == 85


def test_generate_speech_to_video_skips_starting_an_already_running_pod(monkeypatch):
    _fake_ffprobe(monkeypatch)
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    runpod.start_pod("wt5ennaph7w5nh")  # pre-seed as already RUNNING
    runpod.calls.clear()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"58": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}]
    )

    client = SelfHostedS2VVideoClient(runpod, "wt5ennaph7w5nh", comfyui_client_factory=lambda base_url: comfy)
    client.generate_speech_to_video(
        prompt="p", image_data_uri=_data_uri(b"i", "image/jpeg"), audio_data_uri=_data_uri(b"a", "audio/mpeg")
    )

    assert ("start", "wt5ennaph7w5nh") not in runpod.calls


def test_generate_speech_to_video_never_stops_the_pod_itself(monkeypatch):
    # Real design point: this client relies entirely on the on-pod idle
    # watchdog to stop the pod, never an explicit stop here - a version
    # that called stop_pod would restart the pod on every subsequent
    # real line of the same episode.
    _fake_ffprobe(monkeypatch)
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"58": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}]
    )
    client = SelfHostedS2VVideoClient(runpod, "wt5ennaph7w5nh", comfyui_client_factory=lambda base_url: comfy)

    client.generate_speech_to_video(
        prompt="p", image_data_uri=_data_uri(b"i", "image/jpeg"), audio_data_uri=_data_uri(b"a", "audio/mpeg")
    )

    assert ("stop", "wt5ennaph7w5nh") not in runpod.calls


def test_generate_speech_to_video_raises_on_an_unrecognizable_real_output_shape(monkeypatch):
    _fake_ffprobe(monkeypatch)
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(scripted_outputs=[{"58": {"images": []}}])
    client = SelfHostedS2VVideoClient(runpod, "wt5ennaph7w5nh", comfyui_client_factory=lambda base_url: comfy)

    with pytest.raises(SelfHostedS2VError):
        client.generate_speech_to_video(
            prompt="p", image_data_uri=_data_uri(b"i", "image/jpeg"), audio_data_uri=_data_uri(b"a", "audio/mpeg")
        )


def test_self_hosted_s2v_error_is_a_replicate_video_error_for_the_shared_dialogue_pipeline_catch():
    # DialoguePipeline's own except clause is typed around real failures
    # from whichever ReplicateVideoClient is wired in - this must still
    # be catchable the same way.
    assert issubclass(SelfHostedS2VError, ReplicateVideoError)


class _Resp:
    def __init__(self, status_code: int):
        self.status_code = status_code
