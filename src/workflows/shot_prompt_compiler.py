"""The "fast prompt compiler" step: turns each ShotPlan (ShotDirectorAgent,
Gemini) into the two final generation prompts for that shot - one for the
shot's reference still image, one for the speech-to-video model.

Owner's instruction, 2026-10-23: Gemini directs, Groq compiles. Roles
inside this file:
- Groq (the reasoning client passed in) writes the descriptive body of
  each prompt from the character, location, mood, action and shot plan.
- This code (the "engineer" role - plain Python, no third model at
  runtime, per the owner's own choice "Gemini و Groq فقط") guarantees the
  director's decisions survive: the framing, angle, lighting, gaze,
  camera movement and environment motion are written into every prompt
  by code from the ShotPlan itself, never left to Groq to remember, and
  Groq's text is rejected if it puts the character's eyes on the camera
  or leaks Arabic (dialogue) text into a prompt.

Not wired in yet: nothing calls this, and nothing here sends anything to
ComfyUI - DialoguePipeline does that, in a later, separate step.
"""
from __future__ import annotations

import json
import re

from src.agents._text_parsing import parse_json_object_array
from src.agents.shot_director import CAMERA_ANGLES, CAMERA_MOVEMENTS, SHOT_TYPES
from src.integrations.reasoning_client import AgentReasoningClient
from src.models import CompiledShotPrompt, DialogueScreenplay, ShotPlan

SHOT_TYPE_PHRASES = {
    "wide": "wide establishing shot",
    "medium": "medium shot",
    "medium_close_up": "medium close-up shot",
    "close_up": "close-up shot",
    "over_the_shoulder": "over-the-shoulder shot",
    "two_shot": "two-shot",
}
CAMERA_ANGLE_PHRASES = {
    "eye_level": "eye-level camera",
    "low_angle": "low-angle camera",
    "high_angle": "high-angle camera",
    "profile": "profile view",
    "three_quarter": "three-quarter view",
}
CAMERA_MOVEMENT_PHRASES = {
    "static": "static camera",
    "slow_push_in": "slow camera push-in",
    "slow_pull_out": "slow camera pull-out",
    "pan": "slow camera pan",
    "tracking": "tracking camera",
    "handheld": "subtle handheld camera",
}
# Every value the director may return has a phrase here - checked at
# import time so a vocabulary added to shot_director.py can never reach
# a KeyError in production.
assert set(SHOT_TYPE_PHRASES) == set(SHOT_TYPES)
assert set(CAMERA_ANGLE_PHRASES) == set(CAMERA_ANGLES)
assert set(CAMERA_MOVEMENT_PHRASES) == set(CAMERA_MOVEMENTS)

_CAMERA_GAZE_RE = re.compile(
    r"\b(look\w*|star\w*|gaz\w*|fac\w*|eye contact)\s+(directly\s+|straight\s+)?"
    r"((at|into|to|toward|towards|with)\s+)?(the\s+)?(camera|lens|viewer|viewers|audience)\b",
    re.IGNORECASE,
)
_ARABIC_RE = re.compile(r"[؀-ۿ]")

_SYSTEM_PROMPT = """You write prompts for an image model and a speech-to-video model, one shot at a time, for a realistic cinematic drama.
You receive JSON: a list of shots, each with the speaking character (gender, appearance), the scene's location and mood, the stage action, and the director's shot plan.

Reply with ONLY a JSON array, one object per shot, in the same order, each with exactly these keys:
  - "line_index": copied exactly.
  - "image_prompt": one English sentence describing the still frame: the character (from gender and appearance), what they are doing (from action), and the real location around them with concrete set details. Photorealistic film still.
  - "video_prompt": one English sentence describing the motion in this shot: the character's body movement and expression while speaking, and how the scene around them is alive.

Rules:
  - English only. Never include the dialogue, any Arabic, quotation marks or on-screen text.
  - The character never looks at the camera, lens or viewer.
  - Do not mention shot type, camera angle or camera movement - they are added separately.
"""


class ShotPromptCompilerError(Exception):
    pass


class ShotPromptCompiler:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

    def compile(self, screenplay: DialogueScreenplay, shots: list[ShotPlan]) -> list[CompiledShotPrompt]:
        lines = {l.index: l for l in screenplay.lines}
        missing = [s.line_index for s in shots if s.line_index not in lines]
        if not shots or missing:
            raise ShotPromptCompilerError(f"No shots to compile, or shots for unknown lines: {missing}")
        try:
            raw = self._reasoning_client.complete(_SYSTEM_PROMPT, self._user_prompt(screenplay, shots))
        except Exception as exc:
            raise ShotPromptCompilerError(
                f"تعذّر الاتصال بمحرّك الأوامر (Groq) حالياً - حاول مرة أخرى بعد قليل. "
                f"لم يُصرف أي مال حقيقي. السبب التقني: {exc}"
            ) from exc
        items = parse_json_object_array(raw, ("line_index", "image_prompt", "video_prompt"), error_cls=ShotPromptCompilerError)

        expected = [s.line_index for s in shots]
        got = [item.get("line_index") for item in items]
        if got != expected:
            raise ShotPromptCompilerError(f"Compiler returned prompts for lines {got}, needed exactly {expected}")

        compiled: list[CompiledShotPrompt] = []
        for shot, item in zip(shots, items):
            image_body = self._checked(item["image_prompt"], shot.line_index, "image_prompt")
            video_body = self._checked(item["video_prompt"], shot.line_index, "video_prompt")
            compiled.append(
                CompiledShotPrompt(
                    line_index=shot.line_index,
                    image_prompt=(
                        f"{SHOT_TYPE_PHRASES[shot.shot_type]}, {CAMERA_ANGLE_PHRASES[shot.camera_angle]}, "
                        f"{image_body} Lighting: {shot.lighting}. The character looks {shot.gaze}, "
                        f"not at the camera. Photorealistic cinematic film still."
                    ),
                    video_prompt=(
                        f"{CAMERA_MOVEMENT_PHRASES[shot.camera_movement]}, {video_body} "
                        f"In the background: {shot.environment_motion}. The character looks {shot.gaze}."
                    ),
                )
            )
        return compiled

    @staticmethod
    def _user_prompt(screenplay: DialogueScreenplay, shots: list[ShotPlan]) -> str:
        lines = {l.index: l for l in screenplay.lines}
        characters = {c.name: c for c in screenplay.characters}
        scenes = {s.index: s for s in screenplay.scenes}
        payload = []
        for shot in shots:
            line = lines[shot.line_index]
            character = characters.get(line.character)
            scene = scenes.get(line.scene)
            # The dialogue text itself is deliberately NOT sent: the video
            # model lip-syncs the real audio and would render leaked words
            # as extra speech or on-screen text.
            payload.append({
                "line_index": shot.line_index,
                "character": {
                    "gender": character.gender if character else "",
                    "appearance": character.appearance if character else "",
                },
                "location": scene.location if scene else "",
                "mood": scene.mood if scene else "",
                "action": line.action,
                "shot": {
                    "shot_type": shot.shot_type,
                    "camera_angle": shot.camera_angle,
                    "camera_movement": shot.camera_movement,
                    "lighting": shot.lighting,
                    "environment_motion": shot.environment_motion,
                    "gaze": shot.gaze,
                },
            })
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _checked(text: object, line_index: int, key: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ShotPromptCompilerError(f"Line {line_index}: {key} is empty")
        if _CAMERA_GAZE_RE.search(text):
            raise ShotPromptCompilerError(f"Line {line_index}: {key} has the character look at the camera: {text!r}")
        if _ARABIC_RE.search(text):
            raise ShotPromptCompilerError(f"Line {line_index}: {key} contains Arabic text: {text!r}")
        text = text.strip()
        return text if text.endswith(".") else text + "."
