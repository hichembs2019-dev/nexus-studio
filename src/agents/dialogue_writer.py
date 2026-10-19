"""Dialogue Writer Agent: turns a one-line idea into a real character
screenplay - actual people speaking actual lines - the owner's real ask
("مسلسلات وأفلام ... لربح المال", series and films, not a narrated
documentary). Replaces WriterAgent's prose-beat output for any episode
meant to go through DialoguePipeline (real lip-synced speech, see
fal_video_client.py): that pipeline needs a speaker attached to every
line it generates, not a block of narrative text a human would have to
re-parse for "who says what".

Same human gate as WriterAgent: the owner reads/edits
DialogueScreenplay.lines directly before DirectorAgent-equivalent work or
any real generation spends anything. Two LLM calls, not one, because
asking for both the cast list AND every line in a single structured
call regularly produced an inconsistent character roster (a name spelled
two ways is two different reference photos) - cast first, then lines
constrained to that exact roster.

Owner's instruction, 2026-10-19 ("الصور طبعا افتراضية... تُسند تلقائيًا
حسب وصف كل شخصية" - the photos are of course placeholders, assigned
automatically per character description): the cast step now also infers
each character's gender/age_group, and every Character is built with a
real reference_image_path and voice already assigned (see
character_photos.py) - never the empty string DialoguePipeline refuses
to accept.

Real finding, 2026-10-19 ("القصة يجب ان تكون كما طلبتها بالظبط بدون
تحويل" - the story must be exactly as I asked for it, no alteration):
the first real multi-line premise (a real, painful personal story,
already full of directly quoted dialogue in Arabic «guillemets») came
back with invented lines contradicting the premise outright - one
character answered immediately where the premise explicitly says she
stayed silent and answered neither yes nor no, and another was given a
line nobody said in the premise. `_LINES_SYSTEM_PROMPT` previously only
asked for dialogue that "flows" and is "plausible", which gave the model
room to invent freely even when the premise already supplied the real
words. It now explicitly requires reusing a premise's own quoted
dialogue verbatim where one exists, and forbids inventing a line a
character is explicitly described as NOT saying.
"""
from __future__ import annotations

import json

from src.agents._text_parsing import parse_json_object_array
from src.integrations.reasoning_client import AgentReasoningClient
from src.models import Character, DialogueLine, DialogueScreenplay
from src.workflows.character_photos import resolve_placeholder_photo, resolve_placeholder_voice

_CAST_SYSTEM_PROMPT = """You are a casting director listing the speaking characters a short story needs.

Rules:
- Output ONLY a JSON array of objects, nothing else - no markdown fences, no commentary.
- Each object has exactly three keys:
  - "name": the character's name, as it should appear consistently everywhere (same spelling every time).
  - "gender": "male" or "female" - your best real inference from how the premise describes or refers to them.
  - "age_group": "young" (child/teen/20s), "middle" (30s-50s), or "older" (60+) - your best real inference.
- Only characters who actually SPEAK a line - no narrators, no silent background characters.
- 2 to 6 characters, names in the SAME language/script as the premise."""

_LINES_SYSTEM_PROMPT = """You are a screenwriter turning a premise and a fixed cast into a real scripted scene - actual dialogue, not narration.

Rules:
- Output ONLY a JSON array of objects, nothing else - no markdown fences, no commentary.
- The array must have EXACTLY {line_count} elements.
- Each object has exactly three keys:
  - "character": one of these EXACT names (same spelling): {cast}
  - "line": what they actually SAY, in the SAME language as the premise, 1-2 short sentences a person would plausibly speak aloud - never a narrator's description of events.
  - "action": a short, visual-only English stage direction for this moment (facial expression, gesture, what they're doing) - NEVER repeat or translate the spoken line itself here, the video model lip-syncs the real audio and would render duplicated text as more speech.
- The lines must flow as one continuous scene in order, matching the premise, with characters actually responding to each other.
- The premise is the real source of truth, not a loose inspiration: if it already contains a character's actual words (quoted directly, e.g. in "quotes" or «guillemets»), use that EXACT wording for their line - do not paraphrase or invent a different line for a moment the premise already gives you.
- Never give a character a line the premise explicitly says they did NOT say, or have them respond where the premise explicitly says they stayed silent or did not answer - invent dialogue only to fill genuine gaps the premise leaves open, never to contradict what it states."""


class DialogueWriterError(Exception):
    pass


class DialogueWriterAgent:
    def __init__(self, reasoning_client: AgentReasoningClient):
        self._reasoning_client = reasoning_client

    def write_dialogue_screenplay(self, title: str, premise: str, line_count: int) -> DialogueScreenplay:
        if line_count < 1:
            raise DialogueWriterError(f"line_count must be >= 1, got {line_count}")

        cast_raw = self._reasoning_client.complete(_CAST_SYSTEM_PROMPT, premise)
        cast = self._parse_cast(cast_raw)
        cast_names = [c["name"] for c in cast]

        lines_system_prompt = _LINES_SYSTEM_PROMPT.format(line_count=line_count, cast=", ".join(cast_names))
        lines_raw = self._reasoning_client.complete(lines_system_prompt, premise)
        parsed_lines = parse_json_object_array(
            lines_raw, ("character", "line", "action"), error_cls=DialogueWriterError
        )
        if len(parsed_lines) < line_count:
            raise DialogueWriterError(f"Reply returned {len(parsed_lines)} lines, needed {line_count}: {lines_raw!r}")

        for item in parsed_lines[:line_count]:
            if item["character"] not in cast_names:
                raise DialogueWriterError(
                    f"Line speaker {item['character']!r} is not in the cast {cast_names}: {lines_raw!r}"
                )

        characters = [
            Character(
                name=c["name"],
                reference_image_path=resolve_placeholder_photo(c["gender"], c["age_group"]),
                voice=resolve_placeholder_voice(c["gender"]),
            )
            for c in cast
        ]
        dialogue_lines = [
            DialogueLine(index=i, character=item["character"], line=item["line"], action=item["action"])
            for i, item in enumerate(parsed_lines[:line_count])
        ]
        return DialogueScreenplay(title=title, premise=premise, characters=characters, lines=dialogue_lines)

    @staticmethod
    def _parse_cast(raw_reply: str) -> list[dict]:
        text = raw_reply.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DialogueWriterError(f"Cast reply was not valid JSON: {raw_reply!r}") from exc
        if not isinstance(parsed, list) or not parsed:
            raise DialogueWriterError(f"Cast reply was not a non-empty JSON array: {raw_reply!r}")
        for item in parsed:
            if not isinstance(item, dict) or not all(k in item for k in ("name", "gender", "age_group")):
                raise DialogueWriterError(
                    f"Cast reply item {item!r} is missing 'name'/'gender'/'age_group': {raw_reply!r}"
                )
        return parsed
