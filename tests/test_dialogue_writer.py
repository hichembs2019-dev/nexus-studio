from __future__ import annotations

import json

import pytest

from src.agents.dialogue_writer import DialogueWriterAgent, DialogueWriterError
from src.integrations.reasoning_client import MockAgentReasoningClient


def _cast_reply(*people):
    return json.dumps(
        [{"name": n, "gender": g, "age_group": a, "appearance": f"{g}, {a}, ordinary appearance"} for n, g, a in people]
    )


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


def test_write_dialogue_screenplay_raises_if_cast_item_is_missing_appearance():
    # Owner's instruction, 2026-10-20, rejecting a fixed panel of six
    # generic photos reused forever ("لا تكون صور ثابتة الى الابد ...
    # بل حسب القصة و السيناريو"): every cast member needs a real,
    # story-specific appearance description, validated the same way
    # name/gender/age_group already are.
    cast_reply = json.dumps([{"name": "فوزية", "gender": "female", "age_group": "middle"}])  # missing appearance
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="appearance"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_carries_the_real_appearance_onto_the_character():
    cast_reply = json.dumps(
        [{"name": "فوزية", "gender": "female", "age_group": "middle", "appearance": "curly dark hair, slender build"}]
    )
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=1)

    assert screenplay.characters[0].appearance == "curly dark hair, slender build"


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


def test_write_complete_dialogue_screenplay_requires_exactly_one_line_per_real_quoted_moment():
    # Real finding, 2026-10-19 ("يجب ان ينفذ كل حرف و كل كلمة كتبتها
    # بدون حذف او نقصان ... أنا من أدفع المال و أنا حر ماذا أريد" - every
    # letter and word I wrote must be used, no deletion, no shortening -
    # I'm the one paying, I'm free to choose what I want): a mode with NO
    # model-chosen line_count and NO model-chosen selection of which
    # quotes to keep - the real count of quoted moments in the premise
    # drives everything, counted by this code, not guessed by the model.
    premise = 'قال: «مرحبا يا فوزية» فقالت: «أهلا بك» ثم قال: «كيف حالك اليوم؟»'
    cast_reply = _cast_reply(("الرجل", "male", "middle"), ("فوزية", "female", "middle"))
    lines_reply = json.dumps(
        [
            {"character": "الرجل", "line": "مرحبا يا فوزية", "action": "smiling"},
            {"character": "فوزية", "line": "أهلا بك", "action": "smiling back"},
            {"character": "الرجل", "line": "كيف حالك اليوم؟", "action": "leaning in"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_complete_dialogue_screenplay("T", premise)

    assert len(screenplay.lines) == 3
    assert [l.line for l in screenplay.lines] == ["مرحبا يا فوزية", "أهلا بك", "كيف حالك اليوم؟"]
    # the lines prompt must have told the model the real, counted number - 3, not guessed
    assert "exactly 3" in reasoning.calls[1][0]


def test_write_complete_dialogue_screenplay_raises_if_the_model_drops_a_quoted_moment():
    premise = 'قال: «الأولى» وقال: «الثانية» وقال: «الثالثة»'
    cast_reply = _cast_reply(("الرجل", "male", "middle"))
    lines_reply = json.dumps(
        [
            {"character": "الرجل", "line": "الأولى", "action": ""},
            {"character": "الرجل", "line": "الثانية", "action": ""},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="needed 3"):
        writer.write_complete_dialogue_screenplay("T", premise)


def test_write_complete_dialogue_screenplay_prompt_forbids_narrator_and_demands_verbatim_copy():
    premise = 'قال: «مرحبا»'
    cast_reply = _cast_reply(("الرجل", "male", "middle"))
    lines_reply = json.dumps([{"character": "الرجل", "line": "مرحبا", "action": ""}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    writer.write_complete_dialogue_screenplay("T", premise)

    lines_system_prompt = reasoning.calls[1][0]
    assert "narrator" in lines_system_prompt.lower()
    assert "copied EXACTLY" in lines_system_prompt


def test_write_complete_dialogue_screenplay_raises_on_a_premise_with_no_real_quotes():
    # Real bug found by code review, 2026-10-19: max(1, count) used to
    # floor a genuinely quote-free premise at 1, telling the model "the
    # premise contains exactly 1 directly quoted moment" when it
    # contains zero - forcing an invented line, the one thing /full
    # exists to guarantee never happens.
    reasoning = MockAgentReasoningClient(scripted_replies=[])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="لم أجد أي اقتباس"):
        writer.write_complete_dialogue_screenplay("T", "فكرة بلا أي كلام منقول حرفيًا")

    assert reasoning.calls == []  # never even asked the model to invent one


def test_write_complete_dialogue_screenplay_raises_on_an_odd_number_of_straight_quotes():
    # Real incident, 2026-10-20: a real long story's real quote count came
    # back far lower than the real number of quoted moments in it (15
    # instead of 23) with NO error at all - reproduced: one mismatched
    # quote mark anywhere (dropped, duplicated, or mangled by a phone's
    # keyboard while pasting) makes the pattern silently swallow every
    # real line between it and the next closing quote it finds. An odd
    # count of " proves at least one is unpaired somewhere.
    reasoning = MockAgentReasoningClient(scripted_replies=[])
    writer = DialogueWriterAgent(reasoning)

    premise = '"سطر أول" "سطر ثان مفقود علامته الأخيرة "سطر ثالث"'  # 5 straight quotes - odd
    assert premise.count('"') % 2 == 1

    with pytest.raises(DialogueWriterError, match="فردي"):
        writer.write_complete_dialogue_screenplay("T", premise)

    assert reasoning.calls == []  # never risked building from a corrupted count


def test_write_complete_dialogue_screenplay_raises_on_mismatched_curly_quotes():
    reasoning = MockAgentReasoningClient(scripted_replies=[])
    writer = DialogueWriterAgent(reasoning)

    premise = "“سطر أول” “سطر ثان بلا إغلاق"  # one closing ” missing
    with pytest.raises(DialogueWriterError, match="غير متطابقة"):
        writer.write_complete_dialogue_screenplay("T", premise)


def test_write_complete_dialogue_screenplay_raises_on_mismatched_guillemets():
    reasoning = MockAgentReasoningClient(scripted_replies=[])
    writer = DialogueWriterAgent(reasoning)

    premise = "«سطر أول» «سطر ثان بلا إغلاق"  # one closing » missing
    with pytest.raises(DialogueWriterError, match="غير متطابقة"):
        writer.write_complete_dialogue_screenplay("T", premise)


def test_parse_cast_rejects_a_null_field_value():
    # Real bug found by code review, 2026-10-19: the old check only
    # confirmed the keys existed, not that their values were real
    # strings - {"gender": null} passed it and crashed deeper inside
    # character_photos.py's None.strip() instead of raising a clear
    # error here.
    cast_reply = json.dumps([{"name": "X", "gender": None, "age_group": "middle"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="non-string/empty"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_parse_cast_rejects_an_empty_string_field_value():
    cast_reply = json.dumps([{"name": "X", "gender": "  ", "age_group": "middle"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="non-string/empty"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)
