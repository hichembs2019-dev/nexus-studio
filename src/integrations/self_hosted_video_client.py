"""The self-hosted twin of `replicate_video_client.py` - same
`ReplicateVideoClient` interface (one real photo + one real TTS audio
clip in, one real lip-synced mp4 out), run on this project's own Runpod
pod via the real, proven Wan 2.2 S2V graph (`wan_s2v_prompt_builder.py`)
instead of paid per call to Replicate. Drop-in: `DialoguePipeline` takes
this exactly like `RealReplicateVideoClient`, with zero changes to
dialogue_pipeline.py.

Real pod used in production, confirmed live via the Runpod MCP's own
list-pods read 2026-10-22 (not guessed, not assumed from the earlier
proof): pod `wt5ennaph7w5nh`, image
`ghcr.io/hichembs2019-dev/nexus-studio-comfyui:latest` (the custom image
with the idle watchdog baked in - see Dockerfile.comfyui), one NVIDIA
A40, real cost $0.59/hr, status EXITED (not billing compute right now),
mounted on network volume `yamxrkrdos` which already holds every real
Wan 2.2 S2V model file downloaded during the proof run. This is the same
pod docs/SOURCES.md's "Self-hosted Wan 2.2 S2V: a real cinematic clip,
proven end to end" entry used.

Pod lifecycle: `ensure_pod_ready()` (runpod_client.py) is called on
EVERY real call here, not once per episode - cheap when the pod is
already RUNNING and ComfyUI already answers /system_stats (one status
read, one HTTP probe), so a multi-line episode only pays the real
~minutes-long start+readiness cost on its first line. This client never
calls stop_pod() itself - the already-proven on-pod idle watchdog
(scripts/pod_idle_watchdog.py) is the sole, real, end-to-end-verified
stop mechanism for this path, so an episode's real lines can be produced
one after another without a real pod restart between every one of them.

Real cost tracking: DialoguePipeline/BotConversation both compute real
cost as `output_clip_duration_seconds * price_per_second_usd` (a single
shared knob - see dialogue_pipeline.py's own `price_per_second_usd`
docstring). There is no real per-call cost signal this client can report
back through that same interface (ReplicateVideoClient.
generate_speech_to_video returns bytes only), and this project's
standing rule is never to guess a money formula (see the fal.ai 720p/
480p billing-mismatch incident, docs/SOURCES.md). `self_hosted_price_per_second_usd()`
below is therefore built from one real, directly-observed number - the
pod's own real hourly rate - divided into real dollars-per-GPU-second,
and applied to the output clip's own duration as an explicitly
conservative PROXY: a real Wan 2.2 S2V sampling run always takes at
least as long as the clip it produces (loading the model, encoding the
reference photo and audio, 20 sampling steps at cfg 6, decoding, muxing), so
this measure understates the real wall-clock GPU time nearly every real
call actually uses - the same safety-margin posture already used for
Replicate's own PROVISIONAL_PRICE_PER_SECOND_USD. Refining this into a
real measured-wall-clock-time cost is a tracked follow-up once this path
has a real production run to measure, not guessed here.
"""
from __future__ import annotations

import base64
import random
import subprocess
import tempfile
from typing import Callable

from src.integrations.comfyui_s2v_client import ComfyUIS2VClient, RealComfyUIS2VClient
from src.integrations.replicate_video_client import ReplicateVideoClient, ReplicateVideoError
from src.integrations.runpod_client import RunpodClient, ensure_pod_ready
from src.workflows.wan_s2v_prompt_builder import build_s2v_prompt

# Real, live-read rate for pod `wt5ennaph7w5nh` (Runpod MCP list-pods,
# 2026-10-22) - one NVIDIA A40, Secure Cloud, CA-MTL-1. Settings.
# runpod_s2v_hourly_rate_usd defaults to this but is always the real
# knob actually used (see config.py) - never hard-coded past this one
# module-level default, same reasoning dialogue_pipeline.py's own
# price_per_second_usd docstring already gives for Replicate's rate.
DEFAULT_HOURLY_RATE_USD = 0.59

# ComfyUI's own output node ("58" in wan_s2v_prompt_builder.py's graph is
# SaveVideo) reports its real file the same way every Save* node does -
# {"filename", "subfolder", "type"} - under an "images" key regardless of
# the media actually being video, confirmed live against the real S2V
# run's own /history response during the proof.
OUTPUT_NODE_ID = "58"


class SelfHostedS2VError(ReplicateVideoError):
    pass


def _decode_data_uri(data_uri: str) -> tuple[bytes, str]:
    header, _, b64 = data_uri.partition(",")
    if not header.startswith("data:") or ";base64" not in header:
        raise SelfHostedS2VError(f"not a base64 data URI: {data_uri[:60]!r}")
    mime = header[len("data:"):].split(";", 1)[0] or "application/octet-stream"
    return base64.b64decode(b64), mime


def _probe_duration_seconds(audio_bytes: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", tmp.name],
            capture_output=True, text=True,
        )
    if result.returncode != 0 or not result.stdout.strip():
        raise SelfHostedS2VError(f"ffprobe failed on the real TTS audio: {result.stderr[-500:]}")
    return float(result.stdout.strip())


def self_hosted_price_per_second_usd(hourly_rate_usd: float = DEFAULT_HOURLY_RATE_USD) -> float:
    return hourly_rate_usd / 3600.0


class SelfHostedS2VVideoClient(ReplicateVideoClient):
    def __init__(
        self,
        runpod_client: RunpodClient,
        pod_id: str,
        *,
        comfyui_client_factory: Callable[[str], ComfyUIS2VClient] = RealComfyUIS2VClient,
        # Raised from 1800 when production moved to the 20-step, cfg-6
        # path (owner's decision, 2026-10-24): 5x the steps, and cfg > 1
        # adds a second model pass per step. The real per-clip time on
        # this pod has not been measured yet, so the old limit could
        # abandon a clip that was still rendering.
        timeout_seconds: float = 7200,
        poll_interval_seconds: float = 5,
        # 640x640 is the size the real S2V proof ran at (docs/SOURCES.md).
        # The cinematic shot path (app.py) asks for 832x480 instead - the
        # size MultiTalk was proven at on this same pod; NOT yet proven
        # with WanSoundImageToVideo itself.
        width: int = 640,
        height: int = 640,
    ):
        self._width = width
        self._height = height
        self._runpod_client = runpod_client
        self._pod_id = pod_id
        self._comfyui_client_factory = comfyui_client_factory
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def generate_speech_to_video(
        self, *, prompt: str, image_data_uri: str, audio_data_uri: str, resolution: str = "480p"
    ) -> bytes:
        image_bytes, image_mime = _decode_data_uri(image_data_uri)
        audio_bytes, audio_mime = _decode_data_uri(audio_data_uri)
        audio_duration = _probe_duration_seconds(audio_bytes)

        base_url = ensure_pod_ready(self._runpod_client, self._pod_id)
        comfy = self._comfyui_client_factory(base_url)

        ref_name = comfy.upload_file("reference.jpg", image_bytes, image_mime or "image/jpeg")
        audio_name = comfy.upload_file("line.mp3", audio_bytes, audio_mime or "audio/mpeg")

        prompt_body = build_s2v_prompt(
            positive_prompt=prompt,
            ref_image_filename=ref_name,
            audio_filename=audio_name,
            audio_duration_seconds=audio_duration,
            seed=random.randint(0, 2**31 - 1),
            width=self._width,
            height=self._height,
        )
        prompt_id = comfy.submit(prompt_body)
        outputs = comfy.wait_for_completion(
            prompt_id, timeout_seconds=self._timeout_seconds, poll_interval_seconds=self._poll_interval_seconds
        )
        try:
            video_info = outputs[OUTPUT_NODE_ID]["images"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise SelfHostedS2VError(f"ComfyUI S2V run {prompt_id} had no recognizable video output: {outputs!r}") from exc

        return comfy.download_output(
            video_info["filename"], video_info.get("subfolder", ""), video_info.get("type", "output")
        )
