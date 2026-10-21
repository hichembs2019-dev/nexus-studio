"""Shared JSON-reply parsing for the reasoning agents. All ask the model
for strict JSON and all need the same tolerance for a markdown-fenced
reply - kept in one place so a fix to one applies to all.
"""
from __future__ import annotations

import json


def _strip_markdown_fences(raw_reply: str) -> str:
    text = raw_reply.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def parse_json_string_array(raw_reply: str, expected_count: int, *, error_cls: type[Exception]) -> list[str]:
    text = _strip_markdown_fences(raw_reply)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error_cls(f"Reply was not valid JSON: {raw_reply!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise error_cls(f"Reply was not a JSON array of strings: {raw_reply!r}")
    if len(parsed) < expected_count:
        raise error_cls(f"Reply returned {len(parsed)} items, needed {expected_count}: {raw_reply!r}")
    return parsed[:expected_count]


def parse_json_object_array(raw_reply: str, required_keys: tuple[str, ...], *, error_cls: type[Exception]) -> list[dict]:
    """Same tolerance as parse_json_string_array (markdown fences), but for
    a JSON array of objects - used by DialogueWriterAgent, where each beat
    needs a speaker attached to it, not just a bare string."""
    text = _strip_markdown_fences(raw_reply)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error_cls(f"Reply was not valid JSON: {raw_reply!r}") from exc
    if not isinstance(parsed, list):
        raise error_cls(f"Reply was not a JSON array: {raw_reply!r}")
    for item in parsed:
        if not isinstance(item, dict) or not all(key in item for key in required_keys):
            raise error_cls(f"Reply item missing one of {required_keys}: {item!r}")
    return parsed


def parse_json_object(raw_reply: str, required_keys: tuple[str, ...], *, error_cls: type[Exception]) -> dict:
    """Same tolerance as the two above, but for a single JSON object
    reply (not an array) - used by the conversational router, whose
    reply mixes a string ("reply") with lists ("edits"/"caption_lines")."""
    text = _strip_markdown_fences(raw_reply)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error_cls(f"Reply was not valid JSON: {raw_reply!r}") from exc
    if not isinstance(parsed, dict) or not all(key in parsed for key in required_keys):
        raise error_cls(f"Reply missing one of {required_keys}: {raw_reply!r}")
    return parsed
