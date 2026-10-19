from __future__ import annotations

import os
import subprocess

import pytest

from src.integrations.fal_video_client import MockFalVideoClient
from src.integrations.tts_client import MockTTSClient
from src.models import Character, DialogueLine, DialogueScreenplay
from src.workflows.dialogue_pipeline import CINEMATIC_STYLE_SUFFIX, DialoguePipeline, DialoguePipelineError


def _ref_image(path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=16x16", "-frames:v", "1", path],
        capture_output=True, check=True,
    )


def _silent_mp4(path: str, duration_seconds: float) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=16:d={duration_seconds}",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


def test_produce_single_line_rejects_a_character_with_no_reference_image(tmp_path):
    fal_video = MockFalVideoClient()
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path))
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="فوزية", reference_image_path="")],
        lines=[DialogueLine(index=0, character="فوزية", line="مرحبا", action="smiling")],
    )

    with pytest.raises(DialoguePipelineError, match="no reference_image_path"):
        pipeline.produce(screenplay)


def test_produce_single_line_calls_tts_then_fal_with_data_uris_and_saves_the_clip(tmp_path):
    ref_path = str(tmp_path / "fawzia.jpg")
    _ref_image(ref_path)
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient(scripted_bytes=b"\xff\xfbfake-mp3")
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path))
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="فوزية", reference_image_path=ref_path, voice="ar-TN-HediNeural")],
        lines=[DialogueLine(index=0, character="فوزية", line="العشاء جاهز", action="smiling, setting the table")],
    )

    produced = pipeline.produce(screenplay)

    assert tts.calls == [("العشاء جاهز", "ar-TN-HediNeural", "ar-SY")]
    assert len(fal_video.calls) == 1
    call = fal_video.calls[0]
    assert call["prompt"] == "smiling, setting the table" + CINEMATIC_STYLE_SUFFIX
    assert call["image_data_uri"].startswith("data:image/")
    assert call["audio_data_uri"].startswith("data:audio/mpeg;base64,")
    assert call["resolution"] == "480p"

    assert len(produced.clips) == 1
    assert os.path.exists(produced.clips[0].local_path)
    assert produced.stitched_path == produced.clips[0].local_path
    assert produced.clips[0].cost_usd == pytest.approx(1.0 * 0.10)  # 1s clip @ 480p


def test_produce_falls_back_to_a_default_prompt_when_action_is_empty(tmp_path):
    ref_path = str(tmp_path / "ref.jpg")
    _ref_image(ref_path)
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path))
    screenplay = DialogueScreenplay(
        title="T",
        premise="p",
        characters=[Character(name="Jouida", reference_image_path=ref_path)],
        lines=[DialogueLine(index=0, character="Jouida", line="hi", action="")],
    )

    pipeline.produce(screenplay)

    assert fal_video.calls[0]["prompt"] == "Jouida speaking" + CINEMATIC_STYLE_SUFFIX


def test_produce_multi_line_stitches_clips_in_order(tmp_path):
    ref_path = str(tmp_path / "ref.jpg")
    _ref_image(ref_path)
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path))
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="A", reference_image_path=ref_path), Character(name="B", reference_image_path=ref_path)],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="B", line="two", action="b1"),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2
    assert produced.stitched_path is not None
    assert os.path.exists(produced.stitched_path)
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * 0.10)


def test_produce_rejects_a_line_whose_speaker_is_not_in_the_cast(tmp_path):
    pipeline = DialoguePipeline(MockFalVideoClient(), MockTTSClient(), str(tmp_path))
    screenplay = DialogueScreenplay(
        title="T", premise="p", characters=[],
        lines=[DialogueLine(index=0, character="Ghost", line="boo", action="")],
    )

    with pytest.raises(DialoguePipelineError, match="not in the cast"):
        pipeline.produce(screenplay)
