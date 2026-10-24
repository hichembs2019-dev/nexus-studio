"""The "director" step: turns an approved DialogueScreenplay into one real
ShotPlan per line - framing, camera angle, movement, lighting, what moves
in the background, and where the character looks.

Owner's instruction, 2026-10-23: the produced output was a narrator, not a
film ("شخص ثابت يتحدث وكأنه راوي") - Gemini directs the shots, Groq then
compiles them into the final generation prompts (a separate, later step).
Root cause found in code before writing this (dialogue_pipeline.py):
every line of a character reused one cached portrait generated "looking
directly at camera", always tagged "emotional close-up shot". This agent
only plans; nothing consumes its output yet.

Two hard rules are checked in code, not just requested in the prompt -
the same "a code-level gate, not just a prompt request" posture as the
no-narrator gate (docs/SOURCES.md):
- no shot may have the character look at the camera/lens/viewer;
- a screenplay of 3+ lines must not be filmed with one single shot type
  (that IS the talking-head look this exists to remove).
"""
from __future__ import annotations

import json
import re

from src.agents._text_parsing import parse_json_object_array
from src.integrations.reasoning_client import AgentReasoningClient
from src.models import DialogueScreenplay, ShotPlan

SHOT_TYPES = ("wide", "medium", "medium_close_up", "close_up", "over_the_shoulder", "two_shot")
CAMERA_ANGLES = ("eye_level", "low_angle", "high_angle", "profile", "three_quarter")
CAMERA_MOVEMENTS = ("static", "slow_push_in", "slow_pull_out", "pan", "tracking", "handheld")

# Matched as whole words, case-insensitively, against `gaze` only.
_CAMERA_GAZE_WORDS = ("camera", "lens", "viewer", "viewers", "audience")

_SHOT_FIELDS = ("line_index", "shot_type", "camera_angle", "camera_movement", "lighting", "environment_motion", "gaze")

_SYSTEM_PROMPT = f"""You are the director of photography of a realistic cinematic drama.
You receive a screenplay as JSON: characters, scenes (location, mood) and lines (index, character, scene, action, dialogue).
Plan exactly ONE shot per line, so the result plays as a real film scene, never as a person addressing the audience.

Reply with ONLY a JSON array, one object per line, in line order, each with exactly these keys:
  - "line_index": the line's index, copied exactly.
  - "shot_type": one of {list(SHOT_TYPES)}.
  - "camera_angle": one of {list(CAMERA_ANGLES)}.
  - "camera_movement": one of {list(CAMERA_MOVEMENTS)}.
  - "lighting": short English description of the light in this shot (source, direction, color), consistent with the scene's location and mood.
  - "environment_motion": short English description of what visibly moves in the background or environment (people, steam, curtains, traffic, leaves...). Never empty.
  - "gaze": short English description of where the speaking character looks - at the person they talk to, at an object, away, down. NEVER at the camera, lens, viewer or audience.

Rules:
  - Vary the coverage like a real film: establish each new scene with a wider shot, then alternate shot types and angles between lines.
  - The character is always inside the scene's real location, interacting with it - never posed against a plain background.
  - Film the emotion of each line, not just its speaker: read the line's dialogue and the scene's mood, and let the shot carry it - tighter shots and a slow push-in for confession, grief or anger; low-key, high-contrast light for tension; a high angle for a character who is crushed or pleading. A neutral, evenly lit shot is wrong for an emotional line.
  - English only in every value. Never copy or translate the dialogue into any value.
"""


class ShotDirectorError(Exception):
    pass


class ShotDirectorAgent:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

    def plan_shots(self, screenplay: DialogueScreenplay) -> list[ShotPlan]:
        if not screenplay.lines:
            raise ShotDirectorError("السيناريو لا يحتوي على أي سطر حوار لتخطيط لقطاته")
        try:
            raw = self._reasoning_client.complete(_SYSTEM_PROMPT, self._user_prompt(screenplay))
        except Exception as exc:
            raise ShotDirectorError(
                f"تعذّر الاتصال بالمخرج (Gemini) حالياً - حاول مرة أخرى بعد قليل. "
                f"لم يُصرف أي مال حقيقي. السبب التقني: {exc}"
            ) from exc
        items = parse_json_object_array(raw, _SHOT_FIELDS, error_cls=ShotDirectorError)
        return self._validate(items, screenplay)

    @staticmethod
    def _user_prompt(screenplay: DialogueScreenplay) -> str:
        payload = {
            "title": screenplay.title,
            "premise": screenplay.premise,
            "characters": [
                {"name": c.name, "gender": c.gender, "appearance": c.appearance} for c in screenplay.characters
            ],
            "scenes": [{"index": s.index, "location": s.location, "mood": s.mood} for s in screenplay.scenes],
            "lines": [
                {"index": l.index, "character": l.character, "scene": l.scene, "action": l.action, "dialogue": l.line}
                for l in screenplay.lines
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _validate(items: list[dict], screenplay: DialogueScreenplay) -> list[ShotPlan]:
        expected = [l.index for l in screenplay.lines]
        got = [item.get("line_index") for item in items]
        if got != expected:
            raise ShotDirectorError(f"Director returned shots for lines {got}, needed exactly {expected}")

        shots: list[ShotPlan] = []
        for item in items:
            for key, allowed in (
                ("shot_type", SHOT_TYPES),
                ("camera_angle", CAMERA_ANGLES),
                ("camera_movement", CAMERA_MOVEMENTS),
            ):
                if item[key] not in allowed:
                    raise ShotDirectorError(f"Line {item['line_index']}: {key} {item[key]!r} not one of {allowed}")
            for key in ("lighting", "environment_motion", "gaze"):
                if not isinstance(item[key], str) or not item[key].strip():
                    raise ShotDirectorError(f"Line {item['line_index']}: {key} is empty")
            gaze_words = set(re.findall(r"[a-z]+", item["gaze"].lower()))
            if gaze_words & set(_CAMERA_GAZE_WORDS):
                raise ShotDirectorError(f"Line {item['line_index']}: gaze looks at the camera: {item['gaze']!r}")
            shots.append(
                ShotPlan(
                    line_index=item["line_index"],
                    shot_type=item["shot_type"],
                    camera_angle=item["camera_angle"],
                    camera_movement=item["camera_movement"],
                    lighting=item["lighting"].strip(),
                    environment_motion=item["environment_motion"].strip(),
                    gaze=item["gaze"].strip(),
                )
            )

        if len(shots) >= 3 and len({s.shot_type for s in shots}) == 1:
            raise ShotDirectorError(
                f"Director filmed all {len(shots)} lines as {shots[0].shot_type!r} - that is the talking-head look"
            )
        return shots
