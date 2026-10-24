from __future__ import annotations

import json

import pytest

from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import Character, DialogueLine, DialogueSceneInfo, DialogueScreenplay, ShotPlan
from src.workflows.shot_prompt_compiler import ShotPromptCompiler, ShotPromptCompilerError


def _screenplay() -> DialogueScreenplay:
    return DialogueScreenplay(
        title="العشاء",
        premise="فوزية وجويدة في المطبخ",
        characters=[
            Character(name="فوزية", gender="female", appearance="woman in her 50s, grey headscarf"),
            Character(name="جويدة", gender="female", appearance="woman in her 20s, black hair"),
        ],
        lines=[
            DialogueLine(index=0, character="فوزية", line="الرز جاهز", action="lifts the pot lid"),
            DialogueLine(index=1, character="جويدة", line="شكراً يا خالتي", action="sets the table", scene=1),
        ],
        scenes=[
            DialogueSceneInfo(index=0, location="old kitchen", mood="warm"),
            DialogueSceneInfo(index=1, location="dining room", mood="tense"),
        ],
    )


def _shots() -> list[ShotPlan]:
    return [
        ShotPlan(0, "wide", "eye_level", "slow_push_in", "warm window light", "steam rising from the pot", "down at the pot"),
        ShotPlan(1, "close_up", "low_angle", "handheld", "cold ceiling light", "a curtain moving in the wind", "at Fawzia"),
    ]


def _reply(*items: dict) -> str:
    return json.dumps(list(items))


def _item(i: int, image: str = "A woman lifts a steaming pot lid in a cluttered kitchen", video: str = "She speaks softly, shoulders relaxed") -> dict:
    return {"line_index": i, "image_prompt": image, "video_prompt": video}


class _FailingClient:
    def complete(self, system_prompt, user_prompt):
        raise RuntimeError("Groq completion failed: rate limit")


def test_compile_writes_the_directors_decisions_into_every_prompt_by_code():
    client = MockAgentReasoningClient([_reply(_item(0), _item(1))])

    compiled = ShotPromptCompiler(client).compile(_screenplay(), _shots())

    assert [c.line_index for c in compiled] == [0, 1]
    first = compiled[0]
    assert first.image_prompt.startswith("wide establishing shot, eye-level camera, A woman lifts")
    assert "Lighting: warm window light." in first.image_prompt
    assert "looks down at the pot, not at the camera" in first.image_prompt
    assert first.video_prompt.startswith("slow camera push-in, She speaks softly")
    assert "In the background: steam rising from the pot." in first.video_prompt
    assert compiled[1].image_prompt.startswith("close-up shot, low-angle camera,")
    assert compiled[1].video_prompt.startswith("subtle handheld camera,")


def test_groq_gets_character_scene_action_and_shot_but_never_the_dialogue():
    client = MockAgentReasoningClient([_reply(_item(0), _item(1))])

    ShotPromptCompiler(client).compile(_screenplay(), _shots())

    _, user_prompt = client.calls[0]
    payload = json.loads(user_prompt)
    assert payload[1]["location"] == "dining room"
    assert payload[1]["mood"] == "tense"
    assert payload[1]["action"] == "sets the table"
    assert payload[1]["character"]["appearance"] == "woman in her 20s, black hair"
    assert payload[1]["shot"]["shot_type"] == "close_up"
    assert "الرز جاهز" not in user_prompt
    assert "شكراً يا خالتي" not in user_prompt


@pytest.mark.parametrize("text", [
    "She looks directly at the camera while speaking",
    "staring into the lens",
    "facing the viewer",
    "making eye contact with the audience",
])
def test_a_prompt_that_puts_the_eyes_on_the_camera_is_rejected(text):
    client = MockAgentReasoningClient([_reply(_item(0), _item(1, video=text))])

    with pytest.raises(ShotPromptCompilerError, match="look at the camera"):
        ShotPromptCompiler(client).compile(_screenplay(), _shots())


def test_mentioning_the_camera_without_looking_at_it_is_allowed():
    client = MockAgentReasoningClient([_reply(_item(0, image="An old camera sits on the kitchen shelf behind her"), _item(1))])

    assert len(ShotPromptCompiler(client).compile(_screenplay(), _shots())) == 2


def test_arabic_text_leaking_into_a_prompt_is_rejected():
    client = MockAgentReasoningClient([_reply(_item(0, image="She says الرز جاهز"), _item(1))])

    with pytest.raises(ShotPromptCompilerError, match="Arabic"):
        ShotPromptCompiler(client).compile(_screenplay(), _shots())


def test_an_empty_prompt_is_rejected():
    client = MockAgentReasoningClient([_reply(_item(0), _item(1, image=" "))])

    with pytest.raises(ShotPromptCompilerError, match="image_prompt is empty"):
        ShotPromptCompiler(client).compile(_screenplay(), _shots())


def test_prompts_for_the_wrong_lines_are_rejected():
    client = MockAgentReasoningClient([_reply(_item(1), _item(0))])

    with pytest.raises(ShotPromptCompilerError, match="needed exactly"):
        ShotPromptCompiler(client).compile(_screenplay(), _shots())


def test_a_shot_for_a_line_not_in_the_screenplay_is_rejected_without_any_call():
    client = MockAgentReasoningClient([])
    shots = _shots() + [ShotPlan(9, "wide", "eye_level", "static", "x", "y", "z")]

    with pytest.raises(ShotPromptCompilerError, match="unknown lines"):
        ShotPromptCompiler(client).compile(_screenplay(), shots)
    assert client.calls == []


def test_a_provider_failure_becomes_a_clean_arabic_error():
    with pytest.raises(ShotPromptCompilerError, match="تعذّر الاتصال بمحرّك الأوامر"):
        ShotPromptCompiler(_FailingClient()).compile(_screenplay(), _shots())


def test_more_than_8_shots_are_compiled_in_batches():
    screenplay = _screenplay()
    screenplay.lines = [
        DialogueLine(index=i, character="فوزية", line="الرز جاهز", action="lifts the pot lid") for i in range(10)
    ]
    shots = [
        ShotPlan(i, "wide", "eye_level", "static", "warm window light", "steam rising", "down at the pot")
        for i in range(10)
    ]
    client = MockAgentReasoningClient([_reply(*[_item(i) for i in range(8)]), _reply(*[_item(i) for i in range(8, 10)])])

    compiled = ShotPromptCompiler(client).compile(screenplay, shots)

    assert [c.line_index for c in compiled] == list(range(10))
    assert len(client.calls) == 2
    assert [s["line_index"] for s in json.loads(client.calls[1][1])] == [8, 9]
