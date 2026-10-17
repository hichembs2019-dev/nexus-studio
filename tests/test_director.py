from __future__ import annotations

import json

import pytest

from src.agents.director import DirectorAgent, DirectorError
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import CLIP_SECONDS
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
