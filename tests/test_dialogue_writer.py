from __future__ import annotations

import json

import pytest

from src.agents.dialogue_writer import DialogueWriterAgent, DialogueWriterError
from src.integrations.reasoning_client import MockAgentReasoningClient


def test_write_dialogue_screenplay_builds_cast_then_lines():
    cast_reply = json.dumps(["فوزية", "جويدة"])
    lines_reply = json.dumps(
        [
            {"character": "فوزية", "line": "العشاء جاهز يا جويدة", "action": "smiling, setting the table"},
            {"character": "جويدة", "line": "رائحته شهية", "action": "walking in, laughing"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("Family Dinner", "عشاء عائلي عادي", line_count=2)

    assert screenplay.title == "Family Dinner"
    assert [c.name for c in screenplay.characters] == ["فوزية", "جويدة"]
    assert all(c.reference_image_path == "" for c in screenplay.characters)
    assert [l.character for l in screenplay.lines] == ["فوزية", "جويدة"]
    assert screenplay.lines[0].line == "العشاء جاهز يا جويدة"
    assert screenplay.lines[0].action == "smiling, setting the table"
    assert screenplay.lines[0].index == 0

    # the lines call must be constrained to the exact cast from the first call
    assert "فوزية, جويدة" in reasoning.calls[1][0]


def test_write_dialogue_screenplay_raises_if_a_line_speaker_is_not_in_the_cast():
    cast_reply = json.dumps(["فوزية"])
    lines_reply = json.dumps([{"character": "شخص غريب", "line": "مرحبا", "action": "waving"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="not in the cast"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_raises_if_too_few_lines_were_returned():
    cast_reply = json.dumps(["فوزية"])
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="needed 3"):
        writer.write_dialogue_screenplay("T", "p", line_count=3)


def test_write_dialogue_screenplay_raises_on_invalid_cast_json():
    reasoning = MockAgentReasoningClient(scripted_replies=["not json"])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="not valid JSON"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_strips_markdown_fences_on_both_calls():
    cast_reply = "```json\n" + json.dumps(["فوزية"]) + "\n```"
    lines_reply = "```json\n" + json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving"}]) + "\n```"
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=1)

    assert screenplay.lines[0].line == "مرحبا"


def test_write_dialogue_screenplay_rejects_zero_line_count():
    reasoning = MockAgentReasoningClient(scripted_replies=[])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="line_count"):
        writer.write_dialogue_screenplay("T", "p", line_count=0)
