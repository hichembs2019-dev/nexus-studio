"""Turns an approved DialogueScreenplay into real lip-synced clips and one
stitched episode - the dialogue-native counterpart to EpisodePipeline
(episode_pipeline.py), backed by FalVideoClient's real
fal-ai/wan/v2.2-14b/speech-to-video call instead of silent ComfyUI T2V
plus a narration overlay.

Unlike EpisodePipeline, clip length here is NOT fixed (CLIP_SECONDS) - it
is however long that line's own TTS audio runs, because the model lip-
syncs to the real audio track it is given.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
import subprocess

from src.integrations.fal_video_client import PRICE_PER_SECOND_USD, FalVideoClient
from src.integrations.tts_client import TTSClient
from src.models import DialogueScreenplay, ProducedDialogueClip, ProducedDialogueEpisode


class DialoguePipelineError(Exception):
    pass


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "episode"


def _data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime or 'application/octet-stream'};base64,{encoded}"


def _bytes_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr[-500:]}")
    return float(result.stdout.strip())


class DialoguePipeline:
    def __init__(
        self,
        fal_video: FalVideoClient,
        tts: TTSClient,
        output_dir: str,
        *,
        tts_lang: str = "ar-TN",
        resolution: str = "480p",
    ):
        self._fal_video = fal_video
        self._tts = tts
        self._output_dir = output_dir
        self._tts_lang = tts_lang
        self._resolution = resolution

    def produce(self, screenplay: DialogueScreenplay) -> ProducedDialogueEpisode:
        os.makedirs(self._output_dir, exist_ok=True)
        slug = _slug(screenplay.title)
        characters_by_name = {c.name: c for c in screenplay.characters}

        clips: list[ProducedDialogueClip] = []
        for line in screenplay.lines:
            character = characters_by_name.get(line.character)
            if character is None:
                raise DialoguePipelineError(f"Line {line.index} speaker {line.character!r} is not in the cast")
            if not character.reference_image_path:
                raise DialoguePipelineError(
                    f"Character {character.name!r} has no reference_image_path set - "
                    "DialoguePipeline refuses to invent a face (that is exactly the "
                    "'different person every scene' defect this pipeline exists to fix). "
                    "Assign a real reference photo for this character before producing."
                )
            if not os.path.isfile(character.reference_image_path):
                raise DialoguePipelineError(
                    f"Character {character.name!r} reference_image_path "
                    f"{character.reference_image_path!r} does not exist on disk"
                )

            prefix = f"{slug}_line{line.index:03d}"
            audio_bytes = self._tts.synthesize(line.line, character.voice, lang=self._tts_lang)
            audio_path = os.path.join(self._output_dir, f"{prefix}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            image_data_uri = _data_uri(character.reference_image_path)
            audio_data_uri = _bytes_data_uri(audio_bytes, "audio/mpeg")

            clip_bytes = self._fal_video.generate_speech_to_video(
                prompt=line.action or f"{character.name} speaking",
                image_data_uri=image_data_uri,
                audio_data_uri=audio_data_uri,
                resolution=self._resolution,
            )
            clip_path = os.path.join(self._output_dir, f"{prefix}.mp4")
            with open(clip_path, "wb") as f:
                f.write(clip_bytes)

            duration = _ffprobe_duration(clip_path)
            cost = duration * PRICE_PER_SECOND_USD[self._resolution]
            clips.append(ProducedDialogueClip(line=line, local_path=clip_path, cost_usd=cost))

        stitched_path = self._stitch(clips, slug) if len(clips) > 1 else (clips[0].local_path if clips else None)
        return ProducedDialogueEpisode(screenplay=screenplay, clips=clips, stitched_path=stitched_path)

    def _stitch(self, clips: list[ProducedDialogueClip], slug: str) -> str:
        list_path = os.path.join(self._output_dir, f"{slug}_dialogue_concat_list.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(f"file '{os.path.abspath(c.local_path)}'" for c in clips))

        final_path = os.path.join(self._output_dir, f"{slug}_dialogue_stitched.mp4")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-c:a", "aac", final_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg dialogue concat failed: {result.stderr[-2000:]}")
        return final_path
