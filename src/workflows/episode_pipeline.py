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
import subprocess
import time

from src.integrations.comfyui_client import ComfyUIClient
from src.integrations.remote_exec_client import RemoteExecClient
from src.integrations.tts_client import TTSClient
from src.models import Episode, ProducedClip, ProducedEpisode
from src.workflows.wan_prompt_builder import build_ti2v_5b_prompt

COMFYUI_OUTPUT_DIR = "/workspace/runpod-slim/ComfyUI/output"


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "episode"


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr[-500:]}")
    return float(result.stdout.strip())


class EpisodePipeline:
    def __init__(
        self,
        comfyui: ComfyUIClient,
        output_dir: str,
        remote_exec: RemoteExecClient | None = None,
        *,
        tts: TTSClient | None = None,
        tts_voice: str = "ar-TN-HediNeural",
        tts_lang: str = "ar-TN",
        generation_timeout_seconds: float = 1800,
        poll_interval_seconds: float = 5,
    ):
        self._comfyui = comfyui
        self._output_dir = output_dir
        self._remote_exec = remote_exec
        self._tts = tts
        self._tts_voice = tts_voice
        self._tts_lang = tts_lang
        self._generation_timeout_seconds = generation_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def produce(self, episode: Episode, narration_texts: list[str] | None = None) -> ProducedEpisode:
        os.makedirs(self._output_dir, exist_ok=True)
        slug = _slug(episode.title)
        clips: list[ProducedClip] = []
        remote_filenames: list[str] = []
        total_seconds = 0.0
        any_narrated = False

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

            narration_text = narration_texts[scene.index] if narration_texts and scene.index < len(narration_texts) else None
            narrated = False
            if self._tts is not None and narration_text:
                local_path = self._attach_narration(local_path, narration_text, prefix)
                narrated = True

            clips.append(ProducedClip(scene=scene, local_path=local_path, generation_seconds=elapsed))
            any_narrated = any_narrated or narrated

        stitched_path = self._finalize(clips, remote_filenames, slug, any_narrated)
        return ProducedEpisode(episode=episode, clips=clips, stitched_path=stitched_path, total_generation_seconds=total_seconds)

    def _attach_narration(self, video_path: str, text: str, prefix: str) -> str:
        """Overlays a real Azure TTS voiceover onto the clip. If the
        narration runs longer than the 5.04s clip, the video's last frame
        is frozen (ffmpeg tpad) to cover the rest rather than cutting the
        narration short - a silent clip with the right length is a worse
        defect than a few frozen frames at the end of one."""
        audio_bytes = self._tts.synthesize(text, self._tts_voice, lang=self._tts_lang)
        audio_path = os.path.join(self._output_dir, f"{prefix}_narration.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        video_duration = _ffprobe_duration(video_path)
        audio_duration = _ffprobe_duration(audio_path)
        narrated_path = os.path.join(self._output_dir, f"{prefix}_narrated.mp4")

        if audio_duration > video_duration:
            pad_seconds = audio_duration - video_duration + 0.2
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}[v]",
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-c:a", "aac", narrated_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", narrated_path,
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg narration merge failed: {result.stderr[-2000:]}")
        return narrated_path

    def _finalize(self, clips: list[ProducedClip], remote_filenames: list[str], slug: str, any_narrated: bool) -> str | None:
        if len(clips) == 1:
            return clips[0].local_path

        if any_narrated:
            # The remote stitcher concatenates the ORIGINAL silent clips
            # still sitting in ComfyUI's own output dir on the pod - it
            # knows nothing about the narrated copies made locally in
            # _attach_narration(). Using it here would silently stitch the
            # episode back together without the voiceover. Stitch the
            # local, narrated clips instead.
            return self._stitch_local(clips, slug)

        if self._remote_exec is None:
            return None  # multiple clips exist on disk but no stitcher was configured

        final_filename = f"{slug}_stitched.mp4"
        self._stitch_remote(remote_filenames, final_filename)
        final_bytes = self._comfyui.download_output(final_filename, "", "output")
        final_path = os.path.join(self._output_dir, final_filename)
        with open(final_path, "wb") as f:
            f.write(final_bytes)
        return final_path

    def _stitch_local(self, clips: list[ProducedClip], slug: str) -> str:
        list_path = os.path.join(self._output_dir, f"{slug}_concat_list.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(f"file '{os.path.abspath(c.local_path)}'" for c in clips))

        final_path = os.path.join(self._output_dir, f"{slug}_stitched.mp4")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-c:a", "aac", final_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg local concat failed: {result.stderr[-2000:]}")
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
