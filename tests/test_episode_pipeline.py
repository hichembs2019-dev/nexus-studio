from __future__ import annotations

import os

import pytest

from src.integrations.comfyui_client import MockComfyUIClient
from src.integrations.remote_exec_client import MockRemoteExecClient
from src.models import Episode, Scene
from src.workflows.episode_pipeline import COMFYUI_OUTPUT_DIR, EpisodePipeline


def _scene(i: int, text: str) -> Scene:
    return Scene(index=i, positive_prompt=text, negative_prompt="bad quality", seed=i)


def _output_for(filename: str) -> dict:
    return {"58": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}}


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
