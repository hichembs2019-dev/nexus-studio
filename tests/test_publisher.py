from __future__ import annotations

import pytest

from src.agents.publisher import PublisherError, PublishingAgent
from src.integrations.publishing_client import MockPublishingClient
from src.models import Episode, ProducedEpisode


def _produced(stitched_path: str | None = "out/stitched.mp4") -> ProducedEpisode:
    episode = Episode(title="Mars Walk", premise="An astronaut explores Mars.", scenes=[])
    return ProducedEpisode(episode=episode, clips=[], stitched_path=stitched_path, total_generation_seconds=10.0)


def test_publish_episode_uploads_to_every_configured_platform():
    youtube = MockPublishingClient(scripted_video_ids=["yt-id"])
    agent = PublishingAgent({"youtube": youtube})

    results = agent.publish_episode(_produced(), ["youtube"])

    assert len(results) == 1
    assert results[0].platform == "youtube"
    assert results[0].success is True
    assert results[0].video_id == "yt-id"
    assert youtube.uploads[0]["video_path"] == "out/stitched.mp4"
    assert youtube.uploads[0]["title"] == "Mars Walk"
    assert youtube.uploads[0]["description"] == "An astronaut explores Mars."


def test_publish_episode_reports_a_clear_error_for_an_unconfigured_platform():
    agent = PublishingAgent({"youtube": MockPublishingClient()})

    results = agent.publish_episode(_produced(), ["tiktok"])

    assert results[0].platform == "tiktok"
    assert results[0].success is False
    assert "No client configured" in results[0].error


def test_publish_episode_one_platform_failing_does_not_stop_the_others():
    class _FailingClient:
        def upload_video(self, *args, **kwargs):
            raise RuntimeError("upload quota exceeded")

    youtube = MockPublishingClient(scripted_video_ids=["yt-id"])
    agent = PublishingAgent({"youtube": youtube, "broken": _FailingClient()})

    results = agent.publish_episode(_produced(), ["broken", "youtube"])

    assert results[0].success is False
    assert results[0].error == "upload quota exceeded"
    assert results[1].success is True
    assert results[1].video_id == "yt-id"


def test_publish_episode_raises_if_the_episode_was_never_stitched():
    agent = PublishingAgent({"youtube": MockPublishingClient()})

    with pytest.raises(PublisherError, match="no stitched_path"):
        agent.publish_episode(_produced(stitched_path=None), ["youtube"])


def test_publish_episode_uses_overridden_description_and_tags():
    youtube = MockPublishingClient(scripted_video_ids=["yt-id"])
    agent = PublishingAgent({"youtube": youtube})

    agent.publish_episode(_produced(), ["youtube"], description="custom desc", tags=["mars", "scifi"])

    assert youtube.uploads[0]["description"] == "custom desc"
    assert youtube.uploads[0]["tags"] == ["mars", "scifi"]
