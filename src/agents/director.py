"""Director Agent: turns an APPROVED Screenplay (see writer.py) into a
sequence of Scenes, each sized to exactly one Wan 2.2 TI2V-5B generation
call (CLIP_SECONDS, see models.py). It does not generate pixels - that is
EpisodePipeline's job, backed by the comfyui_client that was proven live.

This is deliberately a separate step from writing, with a human gate in
between: the owner writes/edits the Screenplay's scene texts first
(WriterAgent, then the owner's own corrections), and only calls
plan_episode_from_screenplay() once satisfied. Nothing here runs
automatically off a fresh Screenplay.
"""
from __future__ import annotations

from src.agents._text_parsing import parse_json_string_array
from src.integrations.reasoning_client import AgentReasoningClient
from src.models import CLIP_SECONDS, Episode, Scene, Screenplay
from src.workflows.wan_prompt_builder import DEFAULT_NEGATIVE_PROMPT

_SYSTEM_PROMPT = """You are a visual director breaking a story premise into a fixed number of short video clips.

Rules:
- Output ONLY a JSON array of strings, nothing else - no markdown fences, no commentary.
- The array must have EXACTLY {scene_count} elements.
- Each element is one vivid, visual-only scene description for a ~5-second text-to-video clip: concrete subject, setting, action, lighting, camera framing. No dialogue, no sound cues, no scene numbers.
- The scenes must flow as one continuous story in order, matching the premise."""

_SCREENPLAY_SYSTEM_PROMPT = """You are a visual director turning an approved screenplay into shots for a text-to-video model.

Rules:
- Output ONLY a JSON array of strings, nothing else - no markdown fences, no commentary.
- The array must have EXACTLY {scene_count} elements, one per screenplay beat below, IN THE SAME ORDER.
- Each element is a vivid, visual-only scene description IN ENGLISH for a ~5-second text-to-video clip: concrete subject, setting, action, lighting, camera framing. Translate or adapt any dialogue into a visible action instead - the video model cannot render speech or subtitles.
- No dialogue, no sound cues, no scene numbers."""


class DirectorError(Exception):
    pass


class DirectorAgent:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

    def plan_episode(self, title: str, premise: str, target_seconds: float, *, base_seed: int = 1) -> Episode:
        """Shortcut path: premise straight to visual scenes, no screenplay
        in between. Kept for quick tests/prototypes; the owner's real
        workflow is plan_episode_from_screenplay() below."""
        scene_count = max(1, round(target_seconds / CLIP_SECONDS))
        system_prompt = _SYSTEM_PROMPT.format(scene_count=scene_count)
        raw_reply = self._reasoning_client.complete(system_prompt, premise)
        descriptions = parse_json_string_array(raw_reply, scene_count, error_cls=DirectorError)
        return self._build_episode(title, premise, descriptions, base_seed)

    def plan_episode_from_screenplay(self, screenplay: Screenplay, *, base_seed: int = 1) -> Episode:
        scene_count = len(screenplay.scenes)
        system_prompt = _SCREENPLAY_SYSTEM_PROMPT.format(scene_count=scene_count)
        numbered_beats = "\n".join(f"{s.index + 1}. {s.text}" for s in screenplay.scenes)
        raw_reply = self._reasoning_client.complete(system_prompt, numbered_beats)
        descriptions = parse_json_string_array(raw_reply, scene_count, error_cls=DirectorError)
        return self._build_episode(screenplay.title, screenplay.premise, descriptions, base_seed)

    @staticmethod
    def _build_episode(title: str, premise: str, descriptions: list[str], base_seed: int) -> Episode:
        scenes = [
            Scene(index=i, positive_prompt=description, negative_prompt=DEFAULT_NEGATIVE_PROMPT, seed=base_seed + i)
            for i, description in enumerate(descriptions)
        ]
        return Episode(title=title, premise=premise, scenes=scenes)
