from __future__ import annotations

import os
import subprocess

import pytest

from src.integrations.comfyui_client import MockComfyUIClient
from src.integrations.remote_exec_client import MockRemoteExecClient
from src.integrations.tts_client import MockTTSClient
from src.models import Episode, Scene
from src.workflows.episode_pipeline import COMFYUI_OUTPUT_DIR, EpisodePipeline


def _scene(i: int, text: str) -> Scene:
    return Scene(index=i, positive_prompt=text, negative_prompt="bad quality", seed=i)


def _output_for(filename: str) -> dict:
    return {"58": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}}


def _silent_mp4(path: str, duration_seconds: float) -> bytes:
    """A real, playable silent clip - built with ffmpeg itself (same tool
    the pipeline uses), not a byte fixture, so these tests exercise real
    ffprobe/ffmpeg calls just like the pipeline does in production."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=24:d={duration_seconds}",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


def _tone_mp3(path: str, duration_seconds: float) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_seconds}", "-c:a", "libmp3lame", path],
        capture_output=True, check=True,
    )
    with open(path, "rb") as f:
        return f.read()


def _video_stream_frame_count(path: str) -> int:
    """The VIDEO stream's own frame count - not the container's overall
    duration. ffmpeg's default mux already stretches the container to the
    longer of the two input streams even with no frame-freeze filter at
    all (verified live: a 1s video + 3s audio muxed with `-c:v copy -c:a
    aac` alone reports a 3.0s *container* duration but only 24 frames /
    1.0s of actual video). Asserting on container duration alone would
    pass whether or not the clip was really extended - this is why that
    check must read the video stream specifically."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def _has_audio_stream(path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return "audio" in result.stdout


def test_single_scene_episode_is_not_stitched_and_saves_the_one_clip(tmp_path):
    comfyui = MockComfyUIClient(scripted_outputs=[_output_for("ep_scene000.mp4")])
    pipeline = EpisodePipeline(comfyui, str(tmp_path))
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a lone astronaut")])

    produced = pipeline.produce(episode)

    assert len(produced.clips) == 1
    assert produced.stitched_path == produced.clips[0].local_path
    assert os.path.exists(produced.clips[0].local_path)
    with open(produced.clips[0].local_path, "rb") as f:
        assert f.read() == b"fake-mp4-bytes"
    assert comfyui.submitted_prompts[0]["prompt"]["58"]["inputs"]["filename_prefix"] == "ep_scene000"


def test_multi_scene_episode_is_stitched_via_remote_ffmpeg(tmp_path):
    comfyui = MockComfyUIClient(
        scripted_outputs=[_output_for("ep_scene000.mp4"), _output_for("ep_scene001.mp4")],
    )
    remote_exec = MockRemoteExecClient(scripted_outputs=["0\nok"])
    pipeline = EpisodePipeline(comfyui, str(tmp_path), remote_exec)
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a"), _scene(1, "b")])

    produced = pipeline.produce(episode)

    assert len(produced.clips) == 2
    assert produced.stitched_path == os.path.join(str(tmp_path), "ep_stitched.mp4")
    assert os.path.exists(produced.stitched_path)

    ffmpeg_code = remote_exec.commands[0]
    assert f"{COMFYUI_OUTPUT_DIR}/ep_scene000.mp4" in ffmpeg_code
    assert f"{COMFYUI_OUTPUT_DIR}/ep_scene001.mp4" in ffmpeg_code
    assert "ffmpeg" in ffmpeg_code


def test_multi_scene_episode_without_a_remote_exec_leaves_stitched_path_none(tmp_path):
    comfyui = MockComfyUIClient(
        scripted_outputs=[_output_for("ep_scene000.mp4"), _output_for("ep_scene001.mp4")],
    )
    pipeline = EpisodePipeline(comfyui, str(tmp_path), remote_exec=None)
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a"), _scene(1, "b")])

    produced = pipeline.produce(episode)

    assert produced.stitched_path is None
    assert len(produced.clips) == 2  # the clips themselves still exist on disk


def test_total_cost_is_computed_from_real_generation_seconds(tmp_path, monkeypatch):
    times = iter([0.0, 10.0])  # one scene, 10s of wall-clock generation
    monkeypatch.setattr("time.monotonic", lambda: next(times))

    comfyui = MockComfyUIClient(scripted_outputs=[_output_for("ep_scene000.mp4")])
    pipeline = EpisodePipeline(comfyui, str(tmp_path))
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a")])

    produced = pipeline.produce(episode)

    assert produced.total_generation_seconds == 10.0
    assert produced.total_cost_usd == pytest.approx(10.0 * (0.89 / 3600))


def test_single_scene_with_tts_merges_real_narration_onto_the_clip(tmp_path):
    silent_clip = _silent_mp4(str(tmp_path / "src_clip.mp4"), duration_seconds=1.0)
    narration = _tone_mp3(str(tmp_path / "src_audio.mp3"), duration_seconds=0.8)

    comfyui = MockComfyUIClient(scripted_outputs=[_output_for("ep_scene000.mp4")], scripted_bytes=silent_clip)
    tts = MockTTSClient(scripted_bytes=narration)
    pipeline = EpisodePipeline(comfyui, str(tmp_path), tts=tts)
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a lone astronaut")])

    produced = pipeline.produce(episode, narration_texts=["رائد فضاء وحيد"])

    assert tts.calls == [("رائد فضاء وحيد", "ar-TN-HediNeural", "ar-TN")]
    assert produced.clips[0].local_path.endswith("_narrated.mp4")
    assert _has_audio_stream(produced.clips[0].local_path)
    assert produced.stitched_path == produced.clips[0].local_path


def test_narration_longer_than_the_clip_freezes_the_last_frame_instead_of_cutting_it_short(tmp_path):
    # Real finding this guards: a plain audio/video merge truncates to
    # whichever stream is shorter. A narration that overruns a 5s clip
    # must not be cut off mid-sentence - the clip is extended instead.
    silent_clip = _silent_mp4(str(tmp_path / "src_clip.mp4"), duration_seconds=1.0)
    narration = _tone_mp3(str(tmp_path / "src_audio.mp3"), duration_seconds=3.0)

    comfyui = MockComfyUIClient(scripted_outputs=[_output_for("ep_scene000.mp4")], scripted_bytes=silent_clip)
    tts = MockTTSClient(scripted_bytes=narration)
    pipeline = EpisodePipeline(comfyui, str(tmp_path), tts=tts)
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a")])

    produced = pipeline.produce(episode, narration_texts=["a long sentence that runs past the clip"])

    # The clip was shot at 24fps for 1s (24 frames). Without the
    # tpad frame-freeze, the VIDEO stream stays at 24 frames even though
    # the container's overall duration drifts to match the 3s audio
    # track regardless (ffmpeg's default muxing behaviour, not this
    # fix) - so the real frame count is what proves the clip was
    # actually extended to cover the full narration.
    frame_count = _video_stream_frame_count(produced.clips[0].local_path)
    assert frame_count > 24 * 2  # well past the original 1s / 24 frames


def test_no_narration_texts_leaves_the_clip_silent_even_with_tts_configured(tmp_path):
    silent_clip = _silent_mp4(str(tmp_path / "src_clip.mp4"), duration_seconds=1.0)
    comfyui = MockComfyUIClient(scripted_outputs=[_output_for("ep_scene000.mp4")], scripted_bytes=silent_clip)
    tts = MockTTSClient()
    pipeline = EpisodePipeline(comfyui, str(tmp_path), tts=tts)
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a")])

    produced = pipeline.produce(episode, narration_texts=None)

    assert tts.calls == []
    assert not _has_audio_stream(produced.clips[0].local_path)


def test_multi_scene_with_narration_stitches_locally_and_keeps_the_audio(tmp_path):
    # Real bug this guards: the remote stitcher concatenates the ORIGINAL
    # silent clips still sitting on the GPU pod - it knows nothing about
    # the narrated copies made locally. Using it here would silently
    # produce a stitched episode with no voiceover at all.
    # MockComfyUIClient/MockTTSClient return the same scripted bytes on
    # every call, so both scenes get identical (but each individually
    # real and valid) clip/narration content - fine here since the test
    # only checks that the stitched output carries an audio stream.
    clip = _silent_mp4(str(tmp_path / "c0.mp4"), duration_seconds=1.0)
    narration = _tone_mp3(str(tmp_path / "n0.mp3"), duration_seconds=0.8)

    comfyui = MockComfyUIClient(
        scripted_outputs=[_output_for("ep_scene000.mp4"), _output_for("ep_scene001.mp4")],
        scripted_bytes=clip,
    )
    tts = MockTTSClient(scripted_bytes=narration)
    remote_exec = MockRemoteExecClient(scripted_outputs=["0\nok"])
    pipeline = EpisodePipeline(comfyui, str(tmp_path), remote_exec, tts=tts)
    episode = Episode(title="Ep", premise="p", scenes=[_scene(0, "a"), _scene(1, "b")])

    produced = pipeline.produce(episode, narration_texts=["beat one", "beat two"])

    assert remote_exec.commands == []  # the remote (silent) stitcher must never run here
    assert produced.stitched_path is not None
    assert os.path.exists(produced.stitched_path)
    assert _has_audio_stream(produced.stitched_path)
