from __future__ import annotations

import pytest

from src.integrations.comfyui_s2v_client import MockComfyUIS2VClient
from src.integrations.multitalk_video_client import (
    MultiTalkVideoClient,
    MultiTalkVideoError,
)
from src.integrations.runpod_client import MockRunpodClient


def _call(client: MultiTalkVideoClient, **overrides) -> bytes:
    kwargs = dict(
        prompt="two people talking",
        reference_image_bytes=b"ref-bytes",
        reference_image_mime="image/png",
        speaker_1_audio_bytes=b"s1-audio",
        speaker_1_audio_mime="audio/mpeg",
        speaker_1_mask_bytes=b"s1-mask",
        speaker_2_audio_bytes=b"s2-audio",
        speaker_2_audio_mime="audio/mpeg",
        speaker_2_mask_bytes=b"s2-mask",
    )
    kwargs.update(overrides)
    return client.generate_two_speaker_video(**kwargs)


def test_generate_two_speaker_video_ensures_the_pod_is_ready_uploads_builds_and_downloads(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"14": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}],
        scripted_bytes=b"real-multitalk-mp4-bytes",
    )

    client = MultiTalkVideoClient(runpod, "fdhxh9x1igrsih", comfyui_client_factory=lambda base_url: comfy)
    result = _call(client)

    assert result == b"real-multitalk-mp4-bytes"
    assert ("start", "fdhxh9x1igrsih") in runpod.calls  # pod was not already running


def test_generate_two_speaker_video_uploads_every_real_file_once_each(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"14": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}]
    )
    client = MultiTalkVideoClient(runpod, "fdhxh9x1igrsih", comfyui_client_factory=lambda base_url: comfy)
    _call(client)

    uploaded = comfy.uploaded_files
    assert len(uploaded) == 5
    assert uploaded[0] == ("reference.png", b"ref-bytes", "image/png")
    assert uploaded[1] == ("speaker1.mp3", b"s1-audio", "audio/mpeg")
    assert uploaded[2] == ("speaker1_mask.png", b"s1-mask", "image/png")
    assert uploaded[3] == ("speaker2.mp3", b"s2-audio", "audio/mpeg")
    assert uploaded[4] == ("speaker2_mask.png", b"s2-mask", "image/png")


def test_generate_two_speaker_video_wires_the_real_uploaded_names_into_the_submitted_graph(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"14": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}]
    )
    client = MultiTalkVideoClient(runpod, "fdhxh9x1igrsih", comfyui_client_factory=lambda base_url: comfy)
    _call(client, prompt="two people arguing")

    submitted = comfy.submitted_prompts[0]["prompt"]
    assert submitted["8"]["inputs"]["image"] == "reference.png"
    assert submitted["5"]["inputs"]["audio"] == "speaker1.mp3"
    assert submitted["15"]["inputs"]["image"] == "speaker1_mask.png"
    assert submitted["6"]["inputs"]["audio"] == "speaker2.mp3"
    assert submitted["16"]["inputs"]["image"] == "speaker2_mask.png"
    assert submitted["10"]["inputs"]["positive_prompt"] == "two people arguing"


def test_generate_two_speaker_video_skips_starting_an_already_running_pod(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    runpod.start_pod("fdhxh9x1igrsih")
    runpod.calls.clear()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"14": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}]
    )
    client = MultiTalkVideoClient(runpod, "fdhxh9x1igrsih", comfyui_client_factory=lambda base_url: comfy)
    _call(client)

    assert ("start", "fdhxh9x1igrsih") not in runpod.calls


def test_generate_two_speaker_video_never_stops_the_pod_itself(monkeypatch):
    # Same real design point as SelfHostedS2VVideoClient: relies entirely
    # on the on-pod idle watchdog, never an explicit stop here.
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(
        scripted_outputs=[{"14": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}]
    )
    client = MultiTalkVideoClient(runpod, "fdhxh9x1igrsih", comfyui_client_factory=lambda base_url: comfy)
    _call(client)

    assert ("stop", "fdhxh9x1igrsih") not in runpod.calls


def test_generate_two_speaker_video_raises_on_an_unrecognizable_real_output_shape(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))

    runpod = MockRunpodClient()
    comfy = MockComfyUIS2VClient(scripted_outputs=[{"14": {"images": []}}])
    client = MultiTalkVideoClient(runpod, "fdhxh9x1igrsih", comfyui_client_factory=lambda base_url: comfy)

    with pytest.raises(MultiTalkVideoError):
        _call(client)


class _Resp:
    def __init__(self, status_code: int):
        self.status_code = status_code
