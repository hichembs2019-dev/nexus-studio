from __future__ import annotations

import json

import pytest

from src.agents.dialogue_writer import DialogueWriterAgent, DialogueWriterError
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import Character, DialogueLine, DialogueScreenplay


def _cast_reply(*people):
    return json.dumps(
        [{"name": n, "gender": g, "appearance": f"{g}, ordinary appearance"} for n, g in people]
    )


class _AlwaysFailingReasoningClient:
    """Simulates a real provider failure (e.g. Groq AND its Gemini
    fallback both exhausted, a real network error) - something that is
    NOT a DialogueWriterError, the same shape as the real incident,
    2026-10-22: a raw google.genai.errors.ServerError reaching the owner
    verbatim via runner.py's generic top-level handler instead of a
    clean Arabic message through the bot's normal reply path."""

    def complete(self, system_prompt, user_prompt):
        raise RuntimeError("Internal error encountered. {'error': {'code': 500, 'status': 'INTERNAL'}}")


def test_route_message_wraps_a_real_provider_failure_into_a_dialogue_writer_error():
    agent = DialogueWriterAgent(_AlwaysFailingReasoningClient())

    with pytest.raises(DialogueWriterError, match="تعذّر الاتصال بالذكاء الاصطناعي"):
        agent.route_message(None, "", "مرحباً")


def test_write_dialogue_screenplay_wraps_a_real_provider_failure_into_a_dialogue_writer_error():
    agent = DialogueWriterAgent(_AlwaysFailingReasoningClient())

    with pytest.raises(DialogueWriterError, match="تعذّر الاتصال بالذكاء الاصطناعي"):
        agent.write_dialogue_screenplay("T", "premise", 3)


def test_write_complete_dialogue_screenplay_wraps_a_real_provider_failure_into_a_dialogue_writer_error():
    agent = DialogueWriterAgent(_AlwaysFailingReasoningClient())

    with pytest.raises(DialogueWriterError, match="تعذّر الاتصال بالذكاء الاصطناعي"):
        agent.write_complete_dialogue_screenplay("T", 'قال "مرحبا"')


def test_write_dialogue_screenplay_builds_cast_then_lines():
    cast_reply = _cast_reply(("فوزية", "female"), ("جويدة", "female"))
    lines_reply = json.dumps(
        [
            {"character": "فوزية", "line": "العشاء جاهز يا جويدة", "action": "smiling, setting the table", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "جويدة", "line": "رائحته شهية", "action": "walking in, laughing", "scene": 0, "location": "a quiet room", "mood": "calm"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("Family Dinner", "عشاء عائلي عادي", line_count=2)

    assert screenplay.title == "Family Dinner"
    assert [c.name for c in screenplay.characters] == ["فوزية", "جويدة"]
    # Owner's instruction, 2026-10-19: a real voice is always assigned
    # automatically - never the empty string.
    assert all(c.voice for c in screenplay.characters)
    assert [l.character for l in screenplay.lines] == ["فوزية", "جويدة"]
    assert screenplay.lines[0].line == "العشاء جاهز يا جويدة"
    assert screenplay.lines[0].action == "smiling, setting the table"
    assert screenplay.lines[0].index == 0
    # Real scene data, 2026-10-23 - both lines share scene 0, so this is
    # one real DialogueSceneInfo with both speakers present, not two.
    assert screenplay.lines[0].scene == 0
    assert len(screenplay.scenes) == 1
    assert screenplay.scenes[0].location == "a quiet room"
    assert screenplay.scenes[0].characters_present == ["فوزية", "جويدة"]

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
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}])
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
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}])
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
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    writer.write_dialogue_screenplay("T", "p", line_count=1)

    lines_system_prompt = reasoning.calls[1][0]
    assert "turning point" in lines_system_prompt
    assert "WHY something central happens" in lines_system_prompt


def test_write_dialogue_screenplay_raises_if_the_model_casts_a_narrator_anyway():
    # Real incident, 2026-10-21 ("لماذا تصر ان يوجد الراوي في الكودات ...
    # احذف الراوي للابد الا بطلب مني"): the prompt-only fix above (forbid
    # a narrator role) is a REQUEST the model can still ignore - and the
    # real $6.838 wrong video is exactly what that looks like in
    # production. There must be a code-level gate that rejects a
    # narrator-named cast entry outright, not just a prompt asking the
    # model not to add one. This covers the bare name, a name with other
    # words around it, and the English word, all case-insensitively -
    # and never calls the lines step once rejected, so no real cost is
    # risked on a screenplay that was going to be thrown away anyway.
    for narrator_name in ("الراوي", "راوي القصة", "Narrator", "The NARRATOR"):
        cast_reply = _cast_reply(("فوزية", "female"), (narrator_name, "male"))
        reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
        writer = DialogueWriterAgent(reasoning)

        with pytest.raises(DialogueWriterError, match="راوي"):
            writer.write_dialogue_screenplay("T", "p", line_count=1)

        assert len(reasoning.calls) == 1  # the lines step must never be reached


def test_write_dialogue_screenplay_rejects_one_character_dominating_the_lines():
    # Real incident, 2026-10-22: a real produced 22-line episode was, in
    # the owner's own words, "صوت واحد فقط طوال الفيديو" (one voice only,
    # the entire video) - one character carried nearly the whole real
    # screenplay, functionally a narrator even though no character was
    # literally named "راوي". _NARRATOR_NAME_PATTERN never catches this,
    # since it only checks names. 3 of 4 real lines = 75%, over the 70%
    # ceiling.
    cast_reply = _cast_reply(("الشاب", "male"), ("الأب", "male"))
    lines_reply = json.dumps(
        [
            {"character": "الشاب", "line": "أولاً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الشاب", "line": "ثانياً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الشاب", "line": "ثالثاً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الأب", "line": "رابعاً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="الشاب"):
        writer.write_dialogue_screenplay("T", "p", line_count=4)


def test_write_dialogue_screenplay_allows_a_real_imbalance_under_the_ceiling():
    # 2 of 3 lines = 67%, under the 70% ceiling - a real, plausible scene
    # (one character talks more, but the other genuinely participates)
    # must not be rejected for this.
    cast_reply = _cast_reply(("الشاب", "male"), ("الأب", "male"))
    lines_reply = json.dumps(
        [
            {"character": "الشاب", "line": "أولاً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الشاب", "line": "ثانياً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الأب", "line": "ثالثاً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=3)

    assert len(screenplay.lines) == 3


def test_write_dialogue_screenplay_does_not_apply_the_dominance_check_to_a_single_character_cast():
    # A genuinely one-character cast has nothing to "balance" against -
    # not the pattern this guard exists to catch (which is specifically
    # a many-named cast where only one of them ever really speaks).
    cast_reply = _cast_reply(("الشاب", "male"))
    lines_reply = json.dumps(
        [
            {"character": "الشاب", "line": "أولاً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الشاب", "line": "ثانياً", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=2)

    assert len(screenplay.lines) == 2


def test_write_complete_dialogue_screenplay_also_rejects_one_character_dominating_the_lines():
    # Same real guard, same real incident - /full goes through the same
    # _write() method and must be covered identically.
    premise = '«أ» قال «ب» قال «ج» قال «د» رد'
    cast_reply = _cast_reply(("الشاب", "male"), ("الأب", "male"))
    lines_reply = json.dumps(
        [
            {"character": "الشاب", "line": "أ", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الشاب", "line": "ب", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الشاب", "line": "ج", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الأب", "line": "د", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="الشاب"):
        writer.write_complete_dialogue_screenplay("T", premise)


def test_write_dialogue_screenplay_assigns_different_voices_by_gender():
    # Owner's standing instruction, 2026-10-21 ("احذف الستة صور و قلت
    # التوليد حسب القصة" - delete the six photos, generation must be
    # per-story): the cast step no longer assigns any photo at all (no
    # age_group, no placeholder path) - only a voice, by gender. The
    # real, story-specific photo is generated later by DialoguePipeline
    # itself, from `appearance`, never here.
    cast_reply = _cast_reply(("Tom", "male"), ("Amy", "female"))
    lines_reply = json.dumps(
        [
            {"character": "Tom", "line": "Hello", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "Amy", "line": "Hi", "action": "waving back", "scene": 0, "location": "a quiet room", "mood": "calm"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=2)

    tom, amy = screenplay.characters
    assert tom.voice != amy.voice
    assert not hasattr(tom, "reference_image_path")


def test_write_dialogue_screenplay_raises_if_a_line_speaker_is_not_in_the_cast():
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps([{"character": "شخص غريب", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="not in the cast"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_raises_if_too_few_lines_were_returned():
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="needed 3"):
        writer.write_dialogue_screenplay("T", "p", line_count=3)


def test_write_dialogue_screenplay_raises_on_invalid_cast_json():
    reasoning = MockAgentReasoningClient(scripted_replies=["not json"])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="أرسل القصة كاملة"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_raises_if_cast_item_is_missing_a_required_field():
    cast_reply = json.dumps([{"name": "فوزية", "appearance": "ordinary"}])  # missing gender
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="gender"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_raises_if_cast_item_is_missing_appearance():
    # Owner's instruction, 2026-10-20, rejecting a fixed panel of six
    # generic photos reused forever ("لا تكون صور ثابتة الى الابد ...
    # بل حسب القصة و السيناريو"): every cast member needs a real,
    # story-specific appearance description, validated the same way
    # name/gender already are.
    cast_reply = json.dumps([{"name": "فوزية", "gender": "female"}])  # missing appearance
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="appearance"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_carries_the_real_appearance_onto_the_character():
    cast_reply = json.dumps(
        [{"name": "فوزية", "gender": "female", "appearance": "curly dark hair, slender build"}]
    )
    lines_reply = json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=1)

    assert screenplay.characters[0].appearance == "curly dark hair, slender build"


def test_write_dialogue_screenplay_carries_the_real_gender_onto_the_character():
    # Real bug found 2026-10-22, from a real produced episode: gender
    # was inferred here and used ONLY to pick a voice
    # (resolve_placeholder_voice), never stored on Character - so the
    # real portrait prompt (dialogue_pipeline.py) had no gender signal
    # at all, and a character written as "الشاب" (the young man) was
    # given a woman's face. This must now reach the Character object.
    cast_reply = json.dumps([{"name": "أحمد", "gender": "MALE", "appearance": "short hair, broad build"}])
    lines_reply = json.dumps([{"character": "أحمد", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=1)

    # Normalized (lower/stripped) the same way resolve_placeholder_voice
    # already normalizes it, so "MALE" and "male" are never treated as
    # two different real genders.
    assert screenplay.characters[0].gender == "male"


def test_write_dialogue_screenplay_strips_markdown_fences_on_both_calls():
    cast_reply = "```json\n" + _cast_reply(("فوزية", "female")) + "\n```"
    lines_reply = "```json\n" + json.dumps([{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a quiet room", "mood": "calm"}]) + "\n```"
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
    cast_reply = _cast_reply(("الرجل", "male"), ("فوزية", "female"))
    lines_reply = json.dumps(
        [
            {"character": "الرجل", "line": "مرحبا يا فوزية", "action": "smiling", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "فوزية", "line": "أهلا بك", "action": "smiling back", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الرجل", "line": "كيف حالك اليوم؟", "action": "leaning in", "scene": 0, "location": "a quiet room", "mood": "calm"},
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
    cast_reply = _cast_reply(("الرجل", "male"))
    lines_reply = json.dumps(
        [
            {"character": "الرجل", "line": "الأولى", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
            {"character": "الرجل", "line": "الثانية", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="needed 3"):
        writer.write_complete_dialogue_screenplay("T", premise)


def test_write_complete_dialogue_screenplay_prompt_forbids_narrator_and_demands_verbatim_copy():
    premise = 'قال: «مرحبا»'
    cast_reply = _cast_reply(("الرجل", "male"))
    lines_reply = json.dumps([{"character": "الرجل", "line": "مرحبا", "action": "", "scene": 0, "location": "a quiet room", "mood": "calm"}])
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
    cast_reply = json.dumps([{"name": "X", "gender": None, "appearance": "ordinary"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="non-string/empty"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_parse_cast_rejects_an_empty_string_field_value():
    cast_reply = json.dumps([{"name": "X", "gender": "  ", "appearance": "ordinary"}])
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="non-string/empty"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_parse_cast_rejects_two_characters_with_the_same_name():
    # Real bug found by code review, 2026-10-21: DialoguePipeline keys
    # its character lookup by name (a plain dict comprehension), which
    # silently keeps only the LAST of two same-named cast entries -
    # every line attributed to that name would then silently use the
    # wrong character's real generated photo and voice. Never calling
    # the lines step once rejected means no real cost is risked on a
    # screenplay that was always going to be built wrong.
    cast_reply = json.dumps(
        [
            {"name": "الأب", "gender": "male", "appearance": "grey hair, tall"},
            {"name": "الأب", "gender": "male", "appearance": "bald, short"},
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="الأب"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)

    assert len(reasoning.calls) == 1  # the lines step must never be reached


def test_parse_cast_gives_a_clear_arabic_message_when_the_model_replies_with_no_real_json():
    # Real incident, caught live, 2026-10-21: a premise with no real
    # story content (an instruction sentence alone, after the owner's
    # earlier accumulated story was lost to a container restart) made
    # the casting model reply in plain English asking a clarifying
    # question instead of a JSON array. The raw repr of that reply used
    # to reach the owner verbatim and in English - this proves it now
    # gets a clear, actionable Arabic explanation instead.
    reasoning = MockAgentReasoningClient(scripted_replies=["What is the story premise or the characters?"])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="أرسل القصة كاملة"):
        writer.write_dialogue_screenplay("T", "اصنع القصة بالظبط كما طلبت", line_count=1)


def test_parse_cast_gives_a_clear_arabic_message_for_an_empty_cast_array():
    reasoning = MockAgentReasoningClient(scripted_replies=["[]"])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="لم يجد أي شخصية حقيقية"):
        writer.write_dialogue_screenplay("T", "اصنع القصة بالظبط كما طلبت", line_count=1)


# Owner's standing instruction, 2026-10-21 ("اريد الحديث معه كما اتحدث
# معك ... محادثة حرة كاملة") - route_message() tests.


def _screenplay_with_two_lines() -> DialogueScreenplay:
    return DialogueScreenplay(
        title="T", premise="p",
        characters=[Character(name="فوزية", appearance="a"), Character(name="جويدة", appearance="b")],
        lines=[
            DialogueLine(index=0, character="فوزية", line="مرحبا", action="smiling"),
            DialogueLine(index=1, character="جويدة", line="أهلا", action="waving"),
        ],
    )


def test_route_message_accumulate_returns_the_reply_with_no_edits_or_captions():
    router_reply = json.dumps({"intent": "accumulate", "reply": "تم، أكمل القصة.", "edits": [], "caption_lines": []})
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    routed = writer.route_message(None, "", "كان يا ما كان رجل يعيش وحيدًا")

    assert routed.intent == "accumulate"
    assert routed.reply == "تم، أكمل القصة."
    assert routed.edits == []
    assert routed.caption_lines == []


def test_route_message_edit_lines_uses_the_owners_exact_quoted_words():
    screenplay = _screenplay_with_two_lines()
    router_reply = json.dumps(
        {
            "intent": "edit_lines",
            "reply": "عدّلت السطر الأول كما طلبت بالضبط.",
            "edits": [{"line": 1, "text": "لن أسامحك أبدًا"}],
            "caption_lines": [],
        }
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    routed = writer.route_message(screenplay, "", 'اجعل السطر الأول يقول حرفيًا: «لن أسامحك أبدًا»')

    assert routed.intent == "edit_lines"
    assert routed.edits == [(1, "لن أسامحك أبدًا")]


def test_route_message_set_captions_returns_the_real_line_numbers():
    screenplay = _screenplay_with_two_lines()
    router_reply = json.dumps(
        {"intent": "set_captions", "reply": "حوّلت السطر الثاني لنص على الشاشة.", "edits": [], "caption_lines": [2]}
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    routed = writer.route_message(screenplay, "", "اجعل سطر جويدة نصًا مكتوبًا لا صوتًا")

    assert routed.intent == "set_captions"
    assert routed.caption_lines == [2]


def test_route_message_build_screenplay_intent():
    router_reply = json.dumps({"intent": "build_screenplay", "reply": "جارٍ بناء السيناريو الآن.", "edits": [], "caption_lines": []})
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    routed = writer.route_message(None, "قصة طويلة مجمّعة سابقًا", "ابن السيناريو الآن")

    assert routed.intent == "build_screenplay"


def test_route_message_discuss_never_implies_real_production_started():
    # Owner's standing instruction, 2026-10-21: real production must NEVER
    # be triggered by a free message, however it is phrased - only the
    # explicit ✅ button. A message that sounds like approval still comes
    # back as "discuss", and the test covers that the router's own contract
    # allows exactly this - the structural guarantee (never calling
    # DialoguePipeline.produce from here) lives in BotConversation itself.
    router_reply = json.dumps(
        {"intent": "discuss", "reply": "اضغط زر ✅ موافقة الظاهر تحت السيناريو لبدء الإنتاج الحقيقي.", "edits": [], "caption_lines": []}
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    routed = writer.route_message(_screenplay_with_two_lines(), "", "وافقت، انتج الفيديو الآن")

    assert routed.intent == "discuss"
    assert "✅" in routed.reply


def test_router_prompt_requires_answering_the_real_question_not_a_canned_reminder():
    # Real incident, 2026-10-22: a real owner asked direct, unambiguous
    # questions ("اي زر؟", "لماذا لا تجيب يا غبي؟", "ما هي المشكلة في
    # الراوي؟") and got the same boilerplate "press ✅" / unprompted
    # narrator reassurance back every single time, never an answer to
    # what was actually asked - five real replies in a row, none of them
    # addressing the real message. The old prompt only ever told the
    # model what to do when a message "sounds like approval" - nothing
    # constrained the (far more common) case where it does not.
    from src.agents.dialogue_writer import _ROUTER_SYSTEM_PROMPT

    assert "directly and specifically answer" in _ROUTER_SYSTEM_PROMPT
    assert "real failure mode that has happened live" in _ROUTER_SYSTEM_PROMPT


def test_router_prompt_forbids_unprompted_narrator_mentions():
    # Real incident, same evening: every one of those canned replies also
    # brought up "راوٍ" (narrator) unprompted - the owner never asked
    # about one in several of those messages. Traced to the model
    # conflating a caption line's own "(نص على الشاشة، غير منطوق)" marker
    # (route_message()'s own draft_lines rendering) with narration -
    # never explained to the model as anything other than that.
    from src.agents.dialogue_writer import _ROUTER_SYSTEM_PROMPT

    assert "never a narrator" in _ROUTER_SYSTEM_PROMPT
    assert 'unless the owner\'s own new message itself raises it' in _ROUTER_SYSTEM_PROMPT


def test_route_message_rejects_an_unrecognized_intent():
    router_reply = json.dumps({"intent": "produce_now", "reply": "x", "edits": [], "caption_lines": []})
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="unrecognized intent"):
        writer.route_message(None, "", "test")


def test_route_message_rejects_edit_lines_when_no_draft_exists():
    # Real code-level gate, not just a prompt instruction the model could
    # ignore: edit_lines/set_captions are structurally meaningless without
    # an existing draft to edit.
    router_reply = json.dumps(
        {"intent": "edit_lines", "reply": "x", "edits": [{"line": 1, "text": "y"}], "caption_lines": []}
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="no screenplay draft exists"):
        writer.route_message(None, "", "غيّر السطر الأول")


def test_route_message_rejects_edit_lines_with_no_real_edits():
    router_reply = json.dumps({"intent": "edit_lines", "reply": "x", "edits": [], "caption_lines": []})
    reasoning = MockAgentReasoningClient(scripted_replies=[router_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="gave no real edits"):
        writer.route_message(_screenplay_with_two_lines(), "", "غيّر شيئًا")


# Owner's own definition of "فيلم سينمائي", 2026-10-23 ("مشاهد متعددة" -
# multiple scenes) - real multi-scene screenplay tests.


def test_write_dialogue_screenplay_groups_lines_into_real_distinct_scenes():
    # Real, concrete multi-scene case: two scenes, different locations/
    # moods, and a character who only appears in the second one - proves
    # characters_present is derived from the real lines (فوزية never
    # speaks in scene 1 and must not be listed there), not copied blindly
    # from the whole cast.
    cast_reply = _cast_reply(("فوزية", "female"), ("جويدة", "female"), ("الأب", "male"))
    lines_reply = json.dumps(
        [
            {
                "character": "فوزية", "line": "العشاء جاهز", "action": "smiling",
                "scene": 0, "location": "a family kitchen in the evening", "mood": "warm and content",
            },
            {
                "character": "جويدة", "line": "رائحته شهية", "action": "walking in",
                "scene": 0, "location": "a family kitchen in the evening", "mood": "warm and content",
            },
            {
                "character": "جويدة", "line": "أين أبي؟", "action": "stepping outside, worried",
                "scene": 1, "location": "a dark street outside the house", "mood": "tense and uneasy",
            },
            {
                "character": "الأب", "line": "أنا هنا", "action": "emerging from the shadows",
                "scene": 1, "location": "a dark street outside the house", "mood": "tense and uneasy",
            },
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_dialogue_screenplay("T", "p", line_count=4)

    assert [l.scene for l in screenplay.lines] == [0, 0, 1, 1]
    assert len(screenplay.scenes) == 2
    scene0, scene1 = screenplay.scenes
    assert scene0.index == 0
    assert scene0.location == "a family kitchen in the evening"
    assert scene0.mood == "warm and content"
    assert scene0.characters_present == ["فوزية", "جويدة"]
    assert scene1.index == 1
    assert scene1.location == "a dark street outside the house"
    assert scene1.mood == "tense and uneasy"
    # فوزية never speaks in scene 1 - must not appear in its real cast.
    assert scene1.characters_present == ["جويدة", "الأب"]


def test_write_dialogue_screenplay_lines_prompt_asks_for_real_scene_data():
    # Mirrors the narrator/verbatim-quote prompt tests above: proves the
    # real instruction is actually sent to the model, not just that
    # scenes happen to come out right when the mock already supplies them.
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps(
        [{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a room", "mood": "calm"}]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    writer.write_dialogue_screenplay("T", "p", line_count=1)

    lines_system_prompt = reasoning.calls[1][0]
    assert '"scene"' in lines_system_prompt
    assert '"location"' in lines_system_prompt
    assert '"mood"' in lines_system_prompt


def test_write_dialogue_screenplay_rejects_a_non_integer_scene():
    # Mutation check: deleting this validation would let a real screenplay
    # crash deeper (or silently build a wrong scene grouping) instead of
    # failing clearly, right here, before any real cost is spent.
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps(
        [{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": "first", "location": "a room", "mood": "calm"}]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="non-integer or negative 'scene'"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_rejects_a_negative_scene():
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps(
        [{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": -1, "location": "a room", "mood": "calm"}]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="non-integer or negative 'scene'"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_rejects_an_empty_location():
    # Real design point (models.py's own DialogueSceneInfo docstring,
    # restated in this file's module docstring): an empty value here is a
    # real gap in what the model was explicitly asked for, never silently
    # accepted as "not set yet" the way the original additive field was.
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps(
        [{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "  ", "mood": "calm"}]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="real non-empty 'location'"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_dialogue_screenplay_rejects_an_empty_mood():
    cast_reply = _cast_reply(("فوزية", "female"))
    lines_reply = json.dumps(
        [{"character": "فوزية", "line": "مرحبا", "action": "waving", "scene": 0, "location": "a room", "mood": ""}]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    with pytest.raises(DialogueWriterError, match="real non-empty 'mood'"):
        writer.write_dialogue_screenplay("T", "p", line_count=1)


def test_write_complete_dialogue_screenplay_also_groups_real_scenes():
    # Same real grouping, going through /full's own prompt template -
    # must not be a behavior that only works through write_dialogue_screenplay.
    premise = 'قال: «مرحبا يا فوزية» فقالت: «أهلا بك»'
    cast_reply = _cast_reply(("الرجل", "male"), ("فوزية", "female"))
    lines_reply = json.dumps(
        [
            {
                "character": "الرجل", "line": "مرحبا يا فوزية", "action": "smiling",
                "scene": 0, "location": "a sunny garden", "mood": "lighthearted",
            },
            {
                "character": "فوزية", "line": "أهلا بك", "action": "smiling back",
                "scene": 0, "location": "a sunny garden", "mood": "lighthearted",
            },
        ]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[cast_reply, lines_reply])
    writer = DialogueWriterAgent(reasoning)

    screenplay = writer.write_complete_dialogue_screenplay("T", premise)

    assert len(screenplay.scenes) == 1
    assert screenplay.scenes[0].location == "a sunny garden"
    assert screenplay.scenes[0].characters_present == ["الرجل", "فوزية"]


# --- Long screenplays are written in parts (owner's report, 2026-10-24:
# "Reply returned 10 lines, needed 22") ---

def _lines_reply(start: int, count: int, names=("حسين", "ليلى")) -> str:
    return json.dumps([
        {
            "character": names[i % 2], "line": f"سطر {i}", "action": "looks at her", "scene": i // 6,
            "location": f"room {i // 6}", "mood": "tense",
        }
        for i in range(start, start + count)
    ])


def test_a_22_line_screenplay_is_written_in_three_parts_of_8_8_6():
    reasoning = MockAgentReasoningClient([
        _cast_reply(("حسين", "male"), ("ليلى", "female")),
        _lines_reply(0, 8), _lines_reply(8, 8), _lines_reply(16, 6),
    ])

    screenplay = DialogueWriterAgent(reasoning).write_dialogue_screenplay("T", "قصة طويلة", line_count=22)

    assert [l.line for l in screenplay.lines] == [f"سطر {i}" for i in range(22)]
    assert [l.index for l in screenplay.lines] == list(range(22))
    assert [s.index for s in screenplay.scenes] == [0, 1, 2, 3]
    part_calls = reasoning.calls[1:]
    assert len(part_calls) == 3
    assert "PART 1 OF 3" in part_calls[0][0] and "EXACTLY 8 array elements" in part_calls[0][0]
    assert "PART 3 OF 3" in part_calls[2][0] and "EXACTLY 6 array elements" in part_calls[2][0]
    # The first part sees only the premise; later parts also see what was already written.
    assert part_calls[0][1] == "قصة طويلة"
    assert '"سطر 7"' in part_calls[1][1] and '"سطر 8"' not in part_calls[1][1]
    assert '"سطر 15"' in part_calls[2][1]


def test_a_shortened_part_fails_with_which_part_was_short():
    reasoning = MockAgentReasoningClient([
        _cast_reply(("حسين", "male"), ("ليلى", "female")),
        _lines_reply(0, 8), _lines_reply(8, 3),
    ])

    with pytest.raises(DialogueWriterError, match=r"Reply returned 3 lines, needed 8 \(part 2 of 3, lines 9-16 of 22\)"):
        DialogueWriterAgent(reasoning).write_dialogue_screenplay("T", "قصة طويلة", line_count=22)


def test_up_to_8_lines_is_still_one_request():
    reasoning = MockAgentReasoningClient([_cast_reply(("حسين", "male"), ("ليلى", "female")), _lines_reply(0, 8)])

    DialogueWriterAgent(reasoning).write_dialogue_screenplay("T", "قصة", line_count=8)

    assert len(reasoning.calls) == 2
    assert "PART" not in reasoning.calls[1][0]
