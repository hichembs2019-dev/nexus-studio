from __future__ import annotations

import json

import pytest

from src.agents.director import DirectorAgent, DirectorError
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import CLIP_SECONDS, Screenplay, ScreenplayScene
from src.workflows.wan_prompt_builder import DEFAULT_NEGATIVE_PROMPT


def test_plan_episode_sizes_scene_count_from_target_seconds():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["a", "b", "c"])])
    director = DirectorAgent(reasoning)

    episode = director.plan_episode("Mars Walk", "An astronaut explores Mars.", target_seconds=3 * CLIP_SECONDS)

    assert len(episode.scenes) == 3
    assert [s.positive_prompt for s in episode.scenes] == ["a", "b", "c"]
    assert reasoning.calls[0][1] == "An astronaut explores Mars."
    assert "3" in reasoning.calls[0][0]  # system prompt states the exact scene count


def test_plan_episode_assigns_incrementing_seeds_and_default_negative_prompt():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["a", "b"])])
    director = DirectorAgent(reasoning)

    episode = director.plan_episode("T", "p", target_seconds=2 * CLIP_SECONDS, base_seed=100)

    assert [s.seed for s in episode.scenes] == [100, 101]
    assert all(s.negative_prompt == DEFAULT_NEGATIVE_PROMPT for s in episode.scenes)


def test_plan_episode_strips_markdown_code_fences():
    reasoning = MockAgentReasoningClient(scripted_replies=["```json\n" + json.dumps(["only one"]) + "\n```"])
    director = DirectorAgent(reasoning)

    episode = director.plan_episode("T", "p", target_seconds=CLIP_SECONDS)

    assert episode.scenes[0].positive_prompt == "only one"


def test_plan_episode_truncates_extra_scenes():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["a", "b", "c", "d"])])
    director = DirectorAgent(reasoning)

    episode = director.plan_episode("T", "p", target_seconds=2 * CLIP_SECONDS)

    assert [s.positive_prompt for s in episode.scenes] == ["a", "b"]


def test_plan_episode_raises_if_too_few_scenes_were_returned():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["only one"])])
    director = DirectorAgent(reasoning)

    with pytest.raises(DirectorError, match="needed 3"):
        director.plan_episode("T", "p", target_seconds=3 * CLIP_SECONDS)


def test_plan_episode_raises_on_invalid_json():
    reasoning = MockAgentReasoningClient(scripted_replies=["not json at all"])
    director = DirectorAgent(reasoning)

    with pytest.raises(DirectorError, match="not valid JSON"):
        director.plan_episode("T", "p", target_seconds=CLIP_SECONDS)


def test_plan_episode_raises_if_reply_is_not_an_array_of_strings():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps([{"not": "a string"}])])
    director = DirectorAgent(reasoning)

    with pytest.raises(DirectorError, match="not a JSON array of strings"):
        director.plan_episode("T", "p", target_seconds=CLIP_SECONDS)


def test_plan_episode_from_screenplay_sends_the_numbered_beats_and_keeps_order():
    screenplay = Screenplay(
        title="Mars Walk",
        premise="An astronaut explores Mars.",
        scenes=[ScreenplayScene(index=0, text="He steps onto the sand."), ScreenplayScene(index=1, text="He finds a glowing device.")],
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["a boot steps onto red sand", "a glowing device in the sand"])])
    director = DirectorAgent(reasoning)

    episode = director.plan_episode_from_screenplay(screenplay)

    assert episode.title == "Mars Walk"
    assert episode.premise == "An astronaut explores Mars."
    assert [s.positive_prompt for s in episode.scenes] == ["a boot steps onto red sand", "a glowing device in the sand"]
    system_prompt, user_prompt = reasoning.calls[0]
    assert "2" in system_prompt
    assert "1. He steps onto the sand." in user_prompt
    assert "2. He finds a glowing device." in user_prompt


def test_plan_episode_from_screenplay_system_prompt_instructs_carrying_ethnicity():
    # Real finding: Wan 2.2 defaults to East Asian faces when ethnicity is
    # unstated, so a Tunisian/Arab/etc. screenplay rendered East Asian
    # faces until the Director was told to carry that detail into every
    # scene's visual prompt explicitly.
    screenplay = Screenplay(title="T", premise="p", scenes=[ScreenplayScene(index=0, text="a")])
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["a visual shot"])])
    director = DirectorAgent(reasoning)

    director.plan_episode_from_screenplay(screenplay)

    system_prompt, _ = reasoning.calls[0]
    assert "ethnicity" in system_prompt.lower()
    assert "East Asian" in system_prompt


def test_plan_episode_from_screenplay_scene_count_follows_the_screenplay_not_a_fixed_default():
    screenplay = Screenplay(title="T", premise="p", scenes=[ScreenplayScene(index=0, text="only one beat")])
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["one visual shot"])])
    director = DirectorAgent(reasoning)

    episode = director.plan_episode_from_screenplay(screenplay)

    assert len(episode.scenes) == 1


def test_plan_episode_from_screenplay_raises_if_too_few_shots_were_returned():
    screenplay = Screenplay(
        title="T", premise="p", scenes=[ScreenplayScene(index=0, text="a"), ScreenplayScene(index=1, text="b")]
    )
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["only one shot"])])
    director = DirectorAgent(reasoning)

    with pytest.raises(DirectorError, match="needed 2"):
        director.plan_episode_from_screenplay(screenplay)
