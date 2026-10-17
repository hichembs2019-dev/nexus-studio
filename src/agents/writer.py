"""Writer Agent: turns a one-line idea into a full, human-readable
screenplay the owner reads and edits directly - the step the owner asked
for explicitly: "I give the idea, he writes the screenplay, I fix it, and
if I like it I tell the director to direct it."

This is deliberately NOT visual-prompt text (that is DirectorAgent's job,
one step later, on the APPROVED screenplay). A screenplay scene is
written like a short beat: what happens, who's in it, any line of
dialogue - in the SAME language as the premise, since the owner has to
read and correct it before anything is generated.

Scene count still follows CLIP_SECONDS (src/models.py): one Wan 2.2
generation call is a fixed ~5.04s, so however long the episode is meant
to run sets how many beats the screenplay needs - the writer and the
director agree on this number for the same reason.
"""
from __future__ import annotations

from src.agents._text_parsing import parse_json_string_array
from src.integrations.reasoning_client import AgentReasoningClient
from src.models import CLIP_SECONDS, Screenplay, ScreenplayScene

_SYSTEM_PROMPT = """You are a screenwriter turning a premise into a short screenplay for a human to read, edit and approve before any video is made.

Rules:
- Output ONLY a JSON array of strings, nothing else - no markdown fences, no commentary.
- The array must have EXACTLY {scene_count} elements.
- Each element is one screenplay beat: 1-3 sentences, concrete, in the SAME language as the premise. You may include a short line of dialogue in quotes if it fits. No scene numbers, no "INT."/"EXT." sluglines, no camera directions - this is for a human to read and correct, not a shot list.
- The beats must tell one continuous story in order, matching the premise."""


class WriterError(Exception):
    pass


class WriterAgent:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

    def write_screenplay(self, title: str, premise: str, target_seconds: float) -> Screenplay:
        scene_count = max(1, round(target_seconds / CLIP_SECONDS))
        system_prompt = _SYSTEM_PROMPT.format(scene_count=scene_count)
        raw_reply = self._reasoning_client.complete(system_prompt, premise)
        beats = parse_json_string_array(raw_reply, scene_count, error_cls=WriterError)

        scenes = [ScreenplayScene(index=i, text=beat) for i, beat in enumerate(beats)]
        return Screenplay(title=title, premise=premise, scenes=scenes)
