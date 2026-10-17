"""Turns a Director Agent's Episode into real clips and, when there is
more than one scene, one stitched video - backed entirely by the two
real, verified mechanisms in src/integrations: ComfyUIClient (generation,
over HTTPS) and RemoteExecClient (ffmpeg concat on the pod itself, since
every clip already lives there - no need to round-trip bytes through this
process twice).

ComfyUI's own output directory on the proven pod
(/workspace/runpod-slim/ComfyUI/output) is verified live 2026-10-17 - see
docs/SOURCES.md "Wan 2.2 TI2V-5B real generation".
"""
from __future__ import annotations

import os
import re
import time

from src.integrations.comfyui_client import ComfyUIClient
from src.integrations.remote_exec_client import RemoteExecClient
from src.models import Episode, ProducedClip, ProducedEpisode
from src.workflows.wan_prompt_builder import build_ti2v_5b_prompt

COMFYUI_OUTPUT_DIR = "/workspace/runpod-slim/ComfyUI/output"


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "episode"


class EpisodePipeline:
    def __init__(
        self,
        comfyui: ComfyUIClient,
        output_dir: str,
        remote_exec: RemoteExecClient | None = None,
        *,
        generation_timeout_seconds: float = 1800,
        poll_interval_seconds: float = 5,
    ):
        self._comfyui = comfyui
        self._output_dir = output_dir
        self._remote_exec = remote_exec
        self._generation_timeout_seconds = generation_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def produce(self, episode: Episode) -> ProducedEpisode:
        os.makedirs(self._output_dir, exist_ok=True)
        slug = _slug(episode.title)
        clips: list[ProducedClip] = []
        remote_filenames: list[str] = []
        total_seconds = 0.0

        for scene in episode.scenes:
            prefix = f"{slug}_scene{scene.index:03d}"
            prompt_body = build_ti2v_5b_prompt(
                positive_prompt=scene.positive_prompt,
                negative_prompt=scene.negative_prompt,
                seed=scene.seed,
                filename_prefix=prefix,
            )
            start = time.monotonic()
            prompt_id = self._comfyui.submit(prompt_body)
            outputs = self._comfyui.wait_for_completion(
                prompt_id,
                timeout_seconds=self._generation_timeout_seconds,
                poll_interval_seconds=self._poll_interval_seconds,
            )
            elapsed = time.monotonic() - start
            total_seconds += elapsed

            video_info = outputs["58"]["images"][0]
            remote_filenames.append(video_info["filename"])
            clip_bytes = self._comfyui.download_output(video_info["filename"], video_info["subfolder"], video_info["type"])
            local_path = os.path.join(self._output_dir, f"{prefix}.mp4")
            with open(local_path, "wb") as f:
                f.write(clip_bytes)

            clips.append(ProducedClip(scene=scene, local_path=local_path, generation_seconds=elapsed))

        stitched_path = self._finalize(clips, remote_filenames, slug)
        return ProducedEpisode(episode=episode, clips=clips, stitched_path=stitched_path, total_generation_seconds=total_seconds)

    def _finalize(self, clips: list[ProducedClip], remote_filenames: list[str], slug: str) -> str | None:
        if len(clips) == 1:
            return clips[0].local_path
        if self._remote_exec is None:
            return None  # multiple clips exist on disk but no stitcher was configured

        final_filename = f"{slug}_stitched.mp4"
        self._stitch_remote(remote_filenames, final_filename)
        final_bytes = self._comfyui.download_output(final_filename, "", "output")
        final_path = os.path.join(self._output_dir, final_filename)
        with open(final_path, "wb") as f:
            f.write(final_bytes)
        return final_path

    def _stitch_remote(self, remote_filenames: list[str], final_filename: str) -> None:
        list_lines = "\n".join(f"file '{COMFYUI_OUTPUT_DIR}/{name}'" for name in remote_filenames)
        code = f"""
import subprocess
list_path = "{COMFYUI_OUTPUT_DIR}/_concat_list.txt"
with open(list_path, "w") as f:
    f.write({list_lines!r})
result = subprocess.run(
    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "{COMFYUI_OUTPUT_DIR}/{final_filename}"],
    capture_output=True, text=True,
)
print(result.returncode)
print(result.stderr[-2000:])
"""
        output = self._remote_exec.run(code, timeout_seconds=120)
        if not output.strip().startswith("0"):
            raise RuntimeError(f"Remote ffmpeg concat failed: {output}")
