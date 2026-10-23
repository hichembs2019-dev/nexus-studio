from __future__ import annotations

import json

import pytest

from src.agents.shot_director import ShotDirectorAgent, ShotDirectorError
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import Character, DialogueLine, DialogueSceneInfo, DialogueScreenplay, ShotPlan


def _screenplay(n_lines: int = 3) -> DialogueScreenplay:
    return DialogueScreenplay(
        title="العشاء",
        premise="فوزية وجويدة في المطبخ",
        characters=[
            Character(name="فوزية", gender="female", appearance="woman in her 50s"),
            Character(name="جويدة", gender="female", appearance="woman in her 20s"),
        ],
        lines=[
            DialogueLine(index=i, character="فوزية" if i % 2 == 0 else "جويدة", line=f"سطر {i}", action="stirs the pot")
            for i in range(n_lines)
        ],
        scenes=[DialogueSceneInfo(index=0, location="old kitchen", mood="warm")],
    )


def _shot(i: int, shot_type: str = "medium", gaze: str = "at the other woman") -> dict:
    return {
        "line_index": i,
        "shot_type": shot_type,
        "camera_angle": "eye_level",
        "camera_movement": "static",
        "lighting": "warm window light from the left",
        "environment_motion": "steam rising from the pot",
        "gaze": gaze,
    }


def _reply(*shots: dict) -> str:
    return json.dumps(list(shots))


class _FailingClient:
    def complete(self, system_prompt, user_prompt):
        raise RuntimeError("Internal error encountered. {'error': {'code': 500}}")


def test_plan_shots_returns_one_validated_shot_per_line():
    client = MockAgentReasoningClient([_reply(_shot(0, "wide"), _shot(1, "close_up"), _shot(2, "over_the_shoulder"))])

    shots = ShotDirectorAgent(client).plan_shots(_screenplay())

    assert [s.line_index for s in shots] == [0, 1, 2]
    assert shots[0] == ShotPlan(
        line_index=0, shot_type="wide", camera_angle="eye_level", camera_movement="static",
        lighting="warm window light from the left", environment_motion="steam rising from the pot",
        gaze="at the other woman",
    )


def test_the_director_receives_scenes_lines_and_characters_as_json():
    client = MockAgentReasoningClient([_reply(_shot(0, "wide"), _shot(1, "close_up"), _shot(2, "medium"))])

    ShotDirectorAgent(client).plan_shots(_screenplay())

    system_prompt, user_prompt = client.calls[0]
    assert "NEVER at the camera" in system_prompt
    payload = json.loads(user_prompt)
    assert payload["scenes"] == [{"index": 0, "location": "old kitchen", "mood": "warm"}]
    assert [l["index"] for l in payload["lines"]] == [0, 1, 2]
    assert payload["characters"][0]["name"] == "فوزية"


def test_markdown_fenced_reply_is_accepted():
    raw = "```json\n" + _reply(_shot(0, "wide"), _shot(1, "close_up"), _shot(2, "medium")) + "\n```"

    assert len(ShotDirectorAgent(MockAgentReasoningClient([raw])).plan_shots(_screenplay())) == 3


@pytest.mark.parametrize("gaze", ["directly into the camera", "at the lens", "toward the viewer", "at the Audience"])
def test_a_shot_looking_at_the_camera_is_rejected(gaze):
    client = MockAgentReasoningClient([_reply(_shot(0, "wide"), _shot(1, "close_up", gaze=gaze), _shot(2, "medium"))])

    with pytest.raises(ShotDirectorError, match="looks at the camera"):
        ShotDirectorAgent(client).plan_shots(_screenplay())


def test_a_word_that_only_contains_camera_is_not_a_false_positive():
    client = MockAgentReasoningClient(
        [_reply(_shot(0, "wide"), _shot(1, "close_up", gaze="at the cameraman's daughter"), _shot(2, "medium"))]
    )

    assert len(ShotDirectorAgent(client).plan_shots(_screenplay())) == 3


def test_one_single_shot_type_for_the_whole_screenplay_is_rejected():
    client = MockAgentReasoningClient([_reply(_shot(0, "close_up"), _shot(1, "close_up"), _shot(2, "close_up"))])

    with pytest.raises(ShotDirectorError, match="talking-head"):
        ShotDirectorAgent(client).plan_shots(_screenplay())


def test_a_two_line_screenplay_may_share_one_shot_type():
    client = MockAgentReasoningClient([_reply(_shot(0, "medium"), _shot(1, "medium"))])

    assert len(ShotDirectorAgent(client).plan_shots(_screenplay(2))) == 2


def test_missing_or_reordered_lines_are_rejected():
    client = MockAgentReasoningClient([_reply(_shot(0, "wide"), _shot(2, "close_up"), _shot(1, "medium"))])

    with pytest.raises(ShotDirectorError, match="needed exactly"):
        ShotDirectorAgent(client).plan_shots(_screenplay())


@pytest.mark.parametrize("key,value", [("shot_type", "extreme_zoom"), ("camera_angle", "drone"), ("camera_movement", "spin")])
def test_a_value_outside_the_allowed_vocabulary_is_rejected(key, value):
    bad = _shot(1, "close_up")
    bad[key] = value
    client = MockAgentReasoningClient([_reply(_shot(0, "wide"), bad, _shot(2, "medium"))])

    with pytest.raises(ShotDirectorError, match=key):
        ShotDirectorAgent(client).plan_shots(_screenplay())


@pytest.mark.parametrize("key", ["lighting", "environment_motion", "gaze"])
def test_an_empty_description_is_rejected(key):
    bad = _shot(1, "close_up")
    bad[key] = "  "
    client = MockAgentReasoningClient([_reply(_shot(0, "wide"), bad, _shot(2, "medium"))])

    with pytest.raises(ShotDirectorError, match=f"{key} is empty"):
        ShotDirectorAgent(client).plan_shots(_screenplay())


def test_a_provider_failure_becomes_a_clean_arabic_error():
    with pytest.raises(ShotDirectorError, match="تعذّر الاتصال بالمخرج"):
        ShotDirectorAgent(_FailingClient()).plan_shots(_screenplay())


def test_an_empty_screenplay_is_rejected_without_any_call():
    client = MockAgentReasoningClient([])

    with pytest.raises(ShotDirectorError):
        ShotDirectorAgent(client).plan_shots(DialogueScreenplay(title="x", premise="y"))
    assert client.calls == []
