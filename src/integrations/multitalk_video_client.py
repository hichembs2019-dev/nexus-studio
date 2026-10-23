"""The real self-hosted MultiTalk video client: one composite reference
image + two real character face masks + two real TTS audio clips in, one
real video with BOTH characters sharing the same scene out - the real fix
for the structural complaint that started this ("الفرق بينهم في المشهد" -
the difference between them is in the SCENE): SelfHostedS2VVideoClient
produces one isolated clip per character; this produces one real shared
scene for a whole multi-character beat of a screenplay.

Mirrors self_hosted_video_client.py's own shape deliberately (same
ensure_pod_ready/ComfyUIS2VClient reuse, same never-stops-the-pod-itself
reasoning - see that module's own docstring) - this is a second real
caller of the same proven pod-orchestration plumbing, not a new one.

Real, proven live 2026-10-23 (pod `fdhxh9x1igrsih`): build_multitalk_two_
speaker_prompt's exact graph produced a real, playable 832x480 video with
both speakers' mouths independently moving - see docs/SOURCES.md,
"MultiTalk two-speaker generation proven real, end to end". That proof
used a hand-placed reference image/masks from MultiTalk's own public demo
frame, not this project's own generated character portraits - DialoguePipeline's
own `_build_two_speaker_scene_assets` (dialogue_pipeline.py) is what turns
two already-generated real character portraits into the real composite
image + masks this client takes as input. Wiring this client in for a
real production run is a deliberate separate step from building it -
see src/app.py's own comment for the real, explicit config gate.
"""
from __future__ import annotations

import random
from typing import Callable

from src.integrations.comfyui_s2v_client import ComfyUIS2VClient, RealComfyUIS2VClient
from src.integrations.runpod_client import RunpodClient, ensure_pod_ready
from src.workflows.multitalk_prompt_builder import build_multitalk_two_speaker_prompt

OUTPUT_NODE_ID = "14"


class MultiTalkVideoError(Exception):
    pass


class MultiTalkVideoClient:
    def __init__(
        self,
        runpod_client: RunpodClient,
        pod_id: str,
        *,
        comfyui_client_factory: Callable[[str], ComfyUIS2VClient] = RealComfyUIS2VClient,
        timeout_seconds: float = 1800,
        poll_interval_seconds: float = 5,
    ):
        self._runpod_client = runpod_client
        self._pod_id = pod_id
        self._comfyui_client_factory = comfyui_client_factory
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def generate_two_speaker_video(
        self,
        *,
        prompt: str,
        reference_image_bytes: bytes,
        reference_image_mime: str,
        speaker_1_audio_bytes: bytes,
        speaker_1_audio_mime: str,
        speaker_1_mask_bytes: bytes,
        speaker_2_audio_bytes: bytes,
        speaker_2_audio_mime: str,
        speaker_2_mask_bytes: bytes,
    ) -> bytes:
        base_url = ensure_pod_ready(self._runpod_client, self._pod_id)
        comfy = self._comfyui_client_factory(base_url)

        ref_name = comfy.upload_file("reference.png", reference_image_bytes, reference_image_mime or "image/png")
        s1_audio_name = comfy.upload_file("speaker1.mp3", speaker_1_audio_bytes, speaker_1_audio_mime or "audio/mpeg")
        s1_mask_name = comfy.upload_file("speaker1_mask.png", speaker_1_mask_bytes, "image/png")
        s2_audio_name = comfy.upload_file("speaker2.mp3", speaker_2_audio_bytes, speaker_2_audio_mime or "audio/mpeg")
        s2_mask_name = comfy.upload_file("speaker2_mask.png", speaker_2_mask_bytes, "image/png")

        prompt_body = build_multitalk_two_speaker_prompt(
            ref_image_filename=ref_name,
            speaker_1_audio_filename=s1_audio_name,
            speaker_1_mask_filename=s1_mask_name,
            speaker_2_audio_filename=s2_audio_name,
            speaker_2_mask_filename=s2_mask_name,
            positive_prompt=prompt,
            seed=random.randint(0, 2**31 - 1),
        )
        prompt_id = comfy.submit(prompt_body)
        outputs = comfy.wait_for_completion(
            prompt_id, timeout_seconds=self._timeout_seconds, poll_interval_seconds=self._poll_interval_seconds
        )
        try:
            video_info = outputs[OUTPUT_NODE_ID]["images"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise MultiTalkVideoError(f"ComfyUI MultiTalk run {prompt_id} had no recognizable video output: {outputs!r}") from exc

        return comfy.download_output(
            video_info["filename"], video_info.get("subfolder", ""), video_info.get("type", "output")
        )


class MockMultiTalkVideoClient:
    """Scripted mp4 bytes, records every real call - lets a test drive
    DialoguePipeline's real two-speaker pairing logic with no real pod
    and no real money. Mirrors MockReplicateVideoClient's own shape
    (replicate_video_client.py)."""

    def __init__(self, scripted_bytes: bytes = b"fake-multitalk-mp4-bytes", fail_at_calls: frozenset[int] = frozenset()):
        self._scripted_bytes = scripted_bytes
        self._fail_at_calls = fail_at_calls
        self.calls: list[dict] = []

    def generate_two_speaker_video(
        self,
        *,
        prompt: str,
        reference_image_bytes: bytes,
        reference_image_mime: str,
        speaker_1_audio_bytes: bytes,
        speaker_1_audio_mime: str,
        speaker_1_mask_bytes: bytes,
        speaker_2_audio_bytes: bytes,
        speaker_2_audio_mime: str,
        speaker_2_mask_bytes: bytes,
    ) -> bytes:
        call_index = len(self.calls)
        self.calls.append(
            {
                "prompt": prompt,
                "reference_image_bytes": reference_image_bytes,
                "speaker_1_audio_bytes": speaker_1_audio_bytes,
                "speaker_2_audio_bytes": speaker_2_audio_bytes,
            }
        )
        if call_index in self._fail_at_calls:
            raise MultiTalkVideoError(f"mock failure at call {call_index}")
        return self._scripted_bytes
