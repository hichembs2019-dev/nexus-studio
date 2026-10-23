from __future__ import annotations

from src.models import DialogueLine, DialogueSceneInfo, DialogueScreenplay


def test_dialogue_line_defaults_to_scene_zero_for_backward_compatibility():
    # Real design constraint, 2026-10-23: every screenplay produced before
    # real scene support existed - and every existing test fixture in this
    # repository - has no idea scenes exist at all. A line built the old
    # way (no `scene=` kwarg) must land in scene 0, never crash, never
    # require a caller to opt in.
    line = DialogueLine(index=0, character="A", line="hello")
    assert line.scene == 0


def test_dialogue_screenplay_defaults_to_no_scenes_for_backward_compatibility():
    screenplay = DialogueScreenplay(title="T", premise="p")
    assert screenplay.scenes == []


def test_dialogue_scene_info_carries_real_scene_metadata():
    scene = DialogueSceneInfo(
        index=0, location="مطبخ دافئ في بيت العائلة، مساءً",
        characters_present=["الأم", "الابن"], mood="دافئ ومشحون بالتوتر",
    )
    assert scene.index == 0
    assert scene.location == "مطبخ دافئ في بيت العائلة، مساءً"
    assert scene.characters_present == ["الأم", "الابن"]
    assert scene.mood == "دافئ ومشحون بالتوتر"


def test_dialogue_scene_info_mood_defaults_to_empty_not_a_guessed_value():
    # Real design point: a scene with no explicit mood must never silently
    # get an invented one (e.g. "neutral") that a future music step could
    # mistake for a real instruction - empty means "not set", not "calm".
    scene = DialogueSceneInfo(index=0, location="شارع")
    assert scene.mood == ""


def test_a_screenplay_can_group_lines_into_real_distinct_scenes():
    screenplay = DialogueScreenplay(
        title="T", premise="p",
        scenes=[
            DialogueSceneInfo(index=0, location="المطبخ", characters_present=["الأم"], mood="دافئ"),
            DialogueSceneInfo(index=1, location="الشارع", characters_present=["الابن"], mood="متوتر"),
        ],
        lines=[
            DialogueLine(index=0, character="الأم", line="تعال لتأكل", scene=0),
            DialogueLine(index=1, character="الابن", line="لست جائعًا", scene=1),
        ],
    )

    lines_by_scene = {s.index: [l for l in screenplay.lines if l.scene == s.index] for s in screenplay.scenes}

    assert [l.line for l in lines_by_scene[0]] == ["تعال لتأكل"]
    assert [l.line for l in lines_by_scene[1]] == ["لست جائعًا"]
