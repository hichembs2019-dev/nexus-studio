from __future__ import annotations

import json

import pytest

from src.agents.writer import WriterAgent, WriterError
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import CLIP_SECONDS


def test_write_screenplay_sizes_scene_count_from_target_seconds():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["beat one", "beat two", "beat three"])])
    writer = WriterAgent(reasoning)

    screenplay = writer.write_screenplay("Mars Walk", "An astronaut explores Mars.", target_seconds=3 * CLIP_SECONDS)

    assert screenplay.title == "Mars Walk"
    assert screenplay.premise == "An astronaut explores Mars."
    assert [s.text for s in screenplay.scenes] == ["beat one", "beat two", "beat three"]
    assert [s.index for s in screenplay.scenes] == [0, 1, 2]
    assert reasoning.calls[0][1] == "An astronaut explores Mars."
    assert "3" in reasoning.calls[0][0]


def test_write_screenplay_strips_markdown_code_fences():
    reasoning = MockAgentReasoningClient(scripted_replies=["```json\n" + json.dumps(["only one beat"]) + "\n```"])
    writer = WriterAgent(reasoning)

    screenplay = writer.write_screenplay("T", "p", target_seconds=CLIP_SECONDS)

    assert screenplay.scenes[0].text == "only one beat"


def test_write_screenplay_raises_if_too_few_beats_were_returned():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["only one"])])
    writer = WriterAgent(reasoning)

    with pytest.raises(WriterError, match="needed 3"):
        writer.write_screenplay("T", "p", target_seconds=3 * CLIP_SECONDS)


def test_write_screenplay_raises_on_invalid_json():
    reasoning = MockAgentReasoningClient(scripted_replies=["not json"])
    writer = WriterAgent(reasoning)

    with pytest.raises(WriterError, match="not valid JSON"):
        writer.write_screenplay("T", "p", target_seconds=CLIP_SECONDS)
