from __future__ import annotations

import json

import pytest

from src.agents.dialogue_writer import DialogueWriterAgent, DialogueWriterError
from src.integrations.reasoning_client import MockAgentReasoningClient


def _cast_reply(*people):
    return json.dumps([{"name": n, "gender": g, "age_group": a} for n, g, a in people])


def test_write_dialogue_screenplay_builds_cast_then_lines():
    cast_reply = _cast_reply(("فوزية", "female", "middle"), ("جويدة", "female", "young"))
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
    # Owner's instruction, 2026-10-19: real placeholder photos/voices are
    # always assigned automatically - never the empty string
    # DialoguePipeline refuses to accept.
    assert all(c.reference_image_path for c in screenplay.characters)
    assert all(c.reference_image_path.endswith(".png") for c in screenplay.characters)
    assert all(c.voice for c in screenplay.characters)
    assert [l.character for l in screenplay.lines] == ["فوزية", "جويدة"]
    assert screenplay.lines[0].line == "العشاء جاهز يا جويدة"
    assert screenplay.lines[0].action == "smiling, setting the table"
    assert screenplay.lines[0].index == 0

    # the lines call must be constrained to the exact cast from the first call
    assert "فوزية, جويدة" in reasoning.calls[1][0]


def test_write_dialogue_screenplay_lines_prompt_requires_verbatim_quotes_and_forbids_contradiction():
    # Real finding, 2026-10-19: a real story already full of directly
    # quoted dialogue came back with invented lines contradicting it
    # outright (a character answered where the premise says she stayed
    # silent). This checks the instruction that fixes it is actually in
    # the prompt sent to the model, not merely that the method runs -
    # a check that only asserts the call happened would pass with this
    # instruction deleted.
    cast_reply = _cast_reply(("فوزية", "female", "middle"))
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    writer.write_dialogue_screenplay("T", "p", line_count=1)

    lines_system_prompt = reasoning.calls[1][0]
    assert "EXACT wording" in lines_system_prompt
    assert "did NOT say" in lines_system_prompt


def test_write_dialogue_screenplay_prompts_forbid_a_narrator_role():
    # Real finding, 2026-10-19 ("اريد ممثلين يتكلمون لا راوي يروي قصة" -
    # I want actors talking, not a narrator telling a story): a
    # first-person memoir-style premise got its own narrating "I" cast
    # as a character literally named "الراوي" (The Narrator), who was
    # then given lines that summarize past events instead of real
    # in-scene dialogue - the exact documentary-narration failure this
    # whole agent exists to prevent, reappearing through the cast step.
    cast_reply = _cast_reply(("فوزية", "female", "middle"))
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    writer.write_dialogue_screenplay("T", "p", line_count=1)

    cast_system_prompt, lines_system_prompt = reasoning.calls[0][0], reasoning.calls[1][0]
    assert "راوي" in cast_system_prompt and "narrator" in cast_system_prompt.lower()
    assert "DRAMATIZE" in lines_system_prompt


def test_write_dialogue_screenplay_lines_prompt_prioritizes_essential_quoted_moments():
    # Real finding, 2026-10-19 ("انا كتبت قصة و هو يبترها" - I wrote a
    # story and it cuts it): a real 60s/15-line screenplay covered the
    # story's full arc (its last line was the real story's last line)
    # but skipped the single quoted line the whole plot turns on - the
    # mother's lie that caused the divorce - in favor of lines from
    # elsewhere, because the premise had more real quoted moments than
    # line_count allowed and the prompt gave no guidance on which to
    # keep.
    cast_reply = _cast_reply(("فوزية", "female", "middle"))
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    writer.write_dialogue_screenplay("T", "p", line_count=1)

    lines_system_prompt = reasoning.calls[1][0]
    assert "turning point" in lines_system_prompt
    assert "WHY something central happens" in lines_system_prompt


def test_write_dialogue_screenplay_assigns_different_photos_by_gender_and_age():
    cast_reply = _cast_reply(("Tom", "male", "older"), ("Amy", "female", "young"))
    lines_reply = json.dumps(
        [
            {"character": "Tom", "line": "Hello", "action": "waving"},
            {"character": "Amy", "line": "Hi", "action": "waving back"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=2)

    tom, amy = screenplay.characters
    assert tom.reference_image_path != amy.reference_image_path
    assert "male_older" in tom.reference_image_path
    assert "female_young" in amy.reference_image_path
    assert tom.voice != amy.voice


def test_write_dialogue_screenplay_raises_if_a_line_speaker_is_not_in_the_cast():
    cast_reply = _cast_reply(("فوزية", "female", "middle"))
    lines_reply = json.dumps([{"character": "شخص غريب", "line": "مرحبا", "action": "waving"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="not in the cast"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_raises_if_too_few_lines_were_returned():
    cast_reply = _cast_reply(("فوزية", "female", "middle"))
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


def test_write_dialogue_screenplay_raises_if_cast_item_is_missing_a_required_field():
    cast_reply = json.dumps([{"name": "فوزية", "gender": "female"}])  # missing age_group
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="age_group"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_strips_markdown_fences_on_both_calls():
    cast_reply = "```json\n" + _cast_reply(("فوزية", "female", "middle")) + "\n```"
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
