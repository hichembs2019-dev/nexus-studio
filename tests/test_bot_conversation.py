from __future__ import annotations

import json

from src.agents.director import DirectorAgent
from src.agents.publisher import PublishingAgent
from src.agents.writer import WriterAgent
from src.bot.conversation import BotConversation
from src.integrations.publishing_client import MockPublishingClient
from src.integrations.reasoning_client import MockAgentReasoningClient
from src.models import CLIP_SECONDS, Episode, ProducedEpisode, Scene


def _conversation(reasoning_replies: list[str], youtube_video_ids: list[str] | None = None) -> tuple[BotConversation, MockPublishingClient]:
    reasoning = MockAgentReasoningClient(scripted_replies=reasoning_replies)
    youtube = MockPublishingClient(scripted_video_ids=youtube_video_ids or ["yt-id"])
    conversation = BotConversation(WriterAgent(reasoning), DirectorAgent(reasoning), PublishingAgent({"youtube": youtube}))
    return conversation, youtube


def test_new_with_explicit_seconds_writes_a_screenplay_sized_to_them():
    conversation, _ = _conversation([json.dumps(["beat one", "beat two"])])

    reply = conversation.handle_new("chat1", f"{int(2 * CLIP_SECONDS)} A lone astronaut finds a signal.")

    assert "beat one" in reply.text
    assert "beat two" in reply.text
    assert reply.buttons == [("✅ Approve", "approve"), ("❌ Cancel", "cancel")]


def test_new_without_a_leading_number_uses_the_default_duration_and_whole_text_as_premise():
    reasoning = MockAgentReasoningClient(scripted_replies=[json.dumps(["b"] * round(30 / CLIP_SECONDS))])
    youtube = MockPublishingClient()
    conversation = BotConversation(WriterAgent(reasoning), DirectorAgent(reasoning), PublishingAgent({"youtube": youtube}))

    conversation.handle_new("chat1", "A lone astronaut finds a signal.")

    assert reasoning.calls[0][1] == "A lone astronaut finds a signal."


def test_new_with_empty_text_asks_for_an_idea_without_calling_the_writer():
    conversation, _ = _conversation([])

    reply = conversation.handle_new("chat1", "   ")

    assert "Usage" in reply.text


def test_edit_updates_the_named_scene_and_resends_the_screenplay():
    conversation, _ = _conversation([json.dumps(["beat one", "beat two"])])
    conversation.handle_new("chat1", f"{int(2 * CLIP_SECONDS)} idea")

    reply = conversation.handle_edit("chat1", "1 a much better beat")

    assert "a much better beat" in reply.text
    assert "beat one" not in reply.text
    assert "beat two" in reply.text


def test_edit_on_an_out_of_range_scene_reports_it_clearly():
    conversation, _ = _conversation([json.dumps(["beat one"])])
    conversation.handle_new("chat1", f"{int(CLIP_SECONDS)} idea")

    reply = conversation.handle_edit("chat1", "5 something")

    assert "No scene 5" in reply.text


def test_edit_without_a_draft_tells_the_user_to_start_with_new():
    conversation, _ = _conversation([])

    reply = conversation.handle_edit("chat1", "1 x")

    assert "/new" in reply.text


def test_approve_runs_the_director_and_returns_an_episode():
    conversation, _ = _conversation([json.dumps(["beat one"]), json.dumps(["a visual shot"])])
    conversation.handle_new("chat1", f"{int(CLIP_SECONDS)} idea")

    reply, episode = conversation.handle_approve("chat1")

    assert episode is not None
    assert episode.scenes[0].positive_prompt == "a visual shot"
    assert "Generating 1 scene" in reply.text


def test_approve_without_a_draft_returns_none_episode():
    conversation, _ = _conversation([])

    reply, episode = conversation.handle_approve("chat1")

    assert episode is None
    assert "/new" in reply.text


def test_format_produced_offers_publish_buttons_when_stitched():
    conversation, _ = _conversation([])
    episode = Episode(title="T", premise="p", scenes=[Scene(index=0, positive_prompt="x", negative_prompt="y", seed=1)])
    produced = ProducedEpisode(episode=episode, clips=[], stitched_path="/tmp/out.mp4", total_generation_seconds=36.0)

    reply = conversation.format_produced("chat1", produced)

    assert reply.video_path == "/tmp/out.mp4"
    assert reply.buttons == [("📤 Publish to YouTube", "publish:youtube"), ("🗑️ Discard", "discard")]
    assert "0.009" in reply.text  # 36s * 0.89/3600 ≈ 0.0089


def test_format_produced_without_a_stitched_path_reports_cost_without_buttons():
    conversation, _ = _conversation([])
    episode = Episode(title="T", premise="p", scenes=[])
    produced = ProducedEpisode(episode=episode, clips=[], stitched_path=None, total_generation_seconds=10.0)

    reply = conversation.format_produced("chat1", produced)

    assert reply.video_path is None
    assert reply.buttons is None
    assert "could not stitch" in reply.text


def test_publish_youtube_success_reports_the_real_url():
    conversation, youtube = _conversation([], youtube_video_ids=["abc123"])
    episode = Episode(title="T", premise="p", scenes=[])
    produced = ProducedEpisode(episode=episode, clips=[], stitched_path="/tmp/out.mp4", total_generation_seconds=1.0)
    conversation.format_produced("chat1", produced)

    reply = conversation.handle_publish("chat1", "youtube")

    assert reply.text == "✅ Published to Nexora: https://youtu.be/abc123"
    assert youtube.uploads[0]["video_path"] == "/tmp/out.mp4"


def test_publish_with_nothing_produced_reports_that():
    conversation, _ = _conversation([])

    reply = conversation.handle_publish("chat1", "youtube")

    assert "Nothing produced yet" in reply.text


def test_publish_to_unconfigured_platform_reports_the_failure():
    conversation, _ = _conversation([])
    episode = Episode(title="T", premise="p", scenes=[])
    produced = ProducedEpisode(episode=episode, clips=[], stitched_path="/tmp/out.mp4", total_generation_seconds=1.0)
    conversation.format_produced("chat1", produced)

    reply = conversation.handle_publish("chat1", "tiktok")

    assert "failed" in reply.text
    assert "No client configured" in reply.text


def test_cancel_clears_the_draft_so_approve_then_fails():
    conversation, _ = _conversation([json.dumps(["beat one"])])
    conversation.handle_new("chat1", f"{int(CLIP_SECONDS)} idea")

    conversation.handle_cancel("chat1")
    reply, episode = conversation.handle_approve("chat1")

    assert episode is None
    assert "/new" in reply.text


def test_chats_are_independent():
    conversation, _ = _conversation([json.dumps(["chat1 beat"]), json.dumps(["chat2 beat"])])

    conversation.handle_new("chat1", f"{int(CLIP_SECONDS)} idea one")
    conversation.handle_new("chat2", f"{int(CLIP_SECONDS)} idea two")

    reply1 = conversation.handle_edit("chat1", "1 x")
    assert "chat2 beat" not in reply1.text
