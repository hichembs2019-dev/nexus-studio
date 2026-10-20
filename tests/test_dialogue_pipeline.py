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


def test_produce_keeps_real_paid_clips_when_the_final_merge_fails(tmp_path, monkeypatch):
    # Real incident, 2026-10-20: a real run paid for and finished 40.91s
    # of real clips ($8.18, confirmed from fal.ai's own Usage page), then
    # this merge step raised - and because it used to sit outside every
    # try/except, the exception escaped produce() entirely and discarded
    # every one of those real, already-paid, already-on-disk clips. A test
    # that only checks produce() doesn't crash would still pass with the
    # clips thrown away - it must assert the real clips and the real
    # error both survive.
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

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "concat" in cmd:
            class _FailedProcess:
                returncode = 1
                stderr = "simulated real ffmpeg concat failure"
                stdout = ""
            return _FailedProcess()
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2
    assert all(os.path.exists(c.local_path) for c in produced.clips)
    assert produced.stitched_path is None
    assert produced.stitch_error is not None
    assert "simulated real ffmpeg concat failure" in produced.stitch_error
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * 0.10)


def test_produce_stops_spending_once_the_hard_cost_ceiling_is_reached(tmp_path):
    # Owner's instruction, 2026-10-20 ("قلت لك ضع له كود يشتغل و يسحب
    # المال الا عند صنع فيديو" - put code in that only draws money when a
    # video is actually made): max_cost_usd is a hard ceiling on total
    # real spend within one run. A test that only checks the ceiling
    # exists as a number (never compared against anything) would pass
    # with the guard never actually stopping a real fal.ai call - this
    # asserts the THIRD line's real call never happens at all.
    ref_path = str(tmp_path / "ref.jpg")
    _ref_image(ref_path)
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    # Each clip costs 1.0s * $0.10/s (480p) = $0.10. The ceiling is only
    # ever checked BEFORE a line is attempted (its real cost isn't known
    # until after fal.ai actually runs it), so a ceiling below one line's
    # cost is what proves the check fires at all: the first line still
    # goes through (nothing spent yet to compare), every line after it
    # must not.
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path), max_cost_usd=0.05)
    screenplay = DialogueScreenplay(
        title="Dinner",
        premise="p",
        characters=[Character(name="A", reference_image_path=ref_path)],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="A", line="two", action="a2"),
            DialogueLine(index=2, character="A", line="three", action="a3"),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(fal_video.calls) == 1
    assert len(produced.clips) == 1
    assert [l.index for l in produced.skipped_lines] == [1, 2]
    assert produced.total_cost_usd == pytest.approx(0.10)


def test_produce_rejects_a_line_whose_speaker_is_not_in_the_cast(tmp_path):
    pipeline = DialoguePipeline(MockFalVideoClient(), MockTTSClient(), str(tmp_path))
    screenplay = DialogueScreenplay(
        title="T", premise="p", characters=[],
        lines=[DialogueLine(index=0, character="Ghost", line="boo", action="")],
    )

    with pytest.raises(DialoguePipelineError, match="not in the cast"):
        pipeline.produce(screenplay)


def test_produce_keeps_already_paid_for_clips_when_a_later_line_fails(tmp_path):
    # Real finding, 2026-10-19: a real 22-line run paid for and
    # successfully generated ~15 real clips (fal.ai's own dashboard
    # confirmed zero errors on its side), then one line's real job ran
    # past the client's own 600s wait - and produce() used to let that
    # exception propagate straight out, discarding every clip already
    # paid for and already sitting finished on disk. A test that only
    # checks "produce() raises on a fal.ai failure" would still pass
    # with that defect in place - it has to check the earlier real
    # clips survive.
    ref_path = str(tmp_path / "ref.jpg")
    _ref_image(ref_path)
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes, fail_at_calls=frozenset({1}))
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path))
    screenplay = DialogueScreenplay(
        title="T",
        premise="p",
        characters=[Character(name="A", reference_image_path=ref_path)],
        lines=[
            DialogueLine(index=0, character="A", line="one", action="a1"),
            DialogueLine(index=1, character="A", line="two", action="a2"),
            DialogueLine(index=2, character="A", line="three", action="a3"),
        ],
    )

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 2  # line 0 and line 2 succeeded
    assert [c.line.index for c in produced.clips] == [0, 2]
    assert len(produced.failed_lines) == 1
    failed_line, error_message = produced.failed_lines[0]
    assert failed_line.index == 1
    assert "mock failure at call 1" in error_message
    # the fal.ai call itself raised and was never billed for this line -
    # must NOT carry the "real money spent" note
    assert "تم صرف مال حقيقي" not in error_message
    # the real money already spent on the surviving clips must still be
    # reported, not thrown away along with the failed one
    assert produced.total_cost_usd == pytest.approx(2 * 1.0 * 0.10)


def test_produce_reports_when_a_line_was_billed_by_fal_ai_before_the_failure(tmp_path, monkeypatch):
    # Real bug found by code review, 2026-10-19: a single generic
    # failure message used to let the bot claim EVERY failed line's
    # cost "was not counted" - false whenever fal.ai's own real,
    # billed call already succeeded and only a LATER local step (here:
    # ffprobe) failed. A test that only checks a FalVideoError-style
    # message would miss this - it has to check the real billing note
    # actually appears when the paid call itself succeeded.
    ref_path = str(tmp_path / "ref.jpg")
    _ref_image(ref_path)
    clip_bytes = _silent_mp4(str(tmp_path / "src.mp4"), duration_seconds=1.0)

    fal_video = MockFalVideoClient(scripted_bytes=clip_bytes)
    tts = MockTTSClient()
    pipeline = DialoguePipeline(fal_video, tts, str(tmp_path))
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        characters=[Character(name="A", reference_image_path=ref_path)],
        lines=[DialogueLine(index=0, character="A", line="one", action="a1")],
    )

    import src.workflows.dialogue_pipeline as dialogue_pipeline_module

    def _broken_ffprobe(path):
        raise RuntimeError("ffprobe missing")

    monkeypatch.setattr(dialogue_pipeline_module, "_ffprobe_duration", _broken_ffprobe)

    produced = pipeline.produce(screenplay)

    assert len(produced.clips) == 0
    assert len(produced.failed_lines) == 1
    _, error_message = produced.failed_lines[0]
    assert "تم صرف مال حقيقي" in error_message
    assert len(fal_video.calls) == 1  # the real, billed call did happen
