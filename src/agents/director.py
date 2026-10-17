"""Director Agent: turns a one-line premise into a sequence of Scenes,
each sized to exactly one Wan 2.2 TI2V-5B generation call (CLIP_SECONDS,
see models.py). It does not generate pixels - that is EpisodePipeline's
job, backed by the comfyui_client that was proven live.
"""
from __future__ import annotations

import json

from src.integrations.reasoning_client import AgentReasoningClient
from src.models import CLIP_SECONDS, Episode, Scene
from src.workflows.wan_prompt_builder import DEFAULT_NEGATIVE_PROMPT

_SYSTEM_PROMPT = """You are a visual director breaking a story premise into a fixed number of short video clips.

Rules:
- Output ONLY a JSON array of strings, nothing else - no markdown fences, no commentary.
- The array must have EXACTLY {scene_count} elements.
- Each element is one vivid, visual-only scene description for a ~5-second text-to-video clip: concrete subject, setting, action, lighting, camera framing. No dialogue, no sound cues, no scene numbers.
- The scenes must flow as one continuous story in order, matching the premise."""


class DirectorError(Exception):
    pass


class DirectorAgent:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

    def plan_episode(self, title: str, premise: str, target_seconds: float, *, base_seed: int = 1) -> Episode:
        scene_count = max(1, round(target_seconds / CLIP_SECONDS))
        system_prompt = _SYSTEM_PROMPT.format(scene_count=scene_count)
        raw_reply = self._reasoning_client.complete(system_prompt, premise)
        descriptions = self._parse_scene_descriptions(raw_reply, scene_count)

        scenes = [
            Scene(index=i, positive_prompt=description, negative_prompt=DEFAULT_NEGATIVE_PROMPT, seed=base_seed + i)
            for i, description in enumerate(descriptions)
        ]
        return Episode(title=title, premise=premise, scenes=scenes)

    @staticmethod
    def _parse_scene_descriptions(raw_reply: str, scene_count: int) -> list[str]:
        text = raw_reply.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DirectorError(f"Director reply was not valid JSON: {raw_reply!r}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise DirectorError(f"Director reply was not a JSON array of strings: {raw_reply!r}")
        if len(parsed) < scene_count:
            raise DirectorError(f"Director returned {len(parsed)} scenes, needed {scene_count}: {raw_reply!r}")
        return parsed[:scene_count]
