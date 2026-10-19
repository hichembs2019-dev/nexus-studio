"""Publishing Agent: the last step - takes a produced episode's stitched
video and uploads it to whichever configured platforms the owner names.
No reasoning call here (unlike Writer/Director) - title/description come
straight from the Episode/DialogueScreenplay the owner already approved.

Deliberately per-platform fault-tolerant: one platform's failure (missing
client, expired token, a real upload error) does not stop the others -
every platform gets its own PublishResult, success or not.

Shared with both ProducedEpisode (old pipeline) and ProducedDialogueEpisode
(DialogueWriterAgent/DialoguePipeline) - the only difference is whether the
title/premise live on `.episode` or `.screenplay`. Found live, 2026-10-19,
by test_publish_youtube_success_reports_the_real_url after the bot was
rewired to DialoguePipeline: this read produced.episode unconditionally and
raised AttributeError on every real ProducedDialogueEpisode, so the ✅ branch
below the try/except was dead code for the new pipeline and publishing a
real dialogue episode would have silently reported it as a platform
failure instead of actually publishing.
"""
from __future__ import annotations

from src.integrations.publishing_client import PublishingClient
from src.models import ProducedDialogueEpisode, ProducedEpisode, PublishResult


class PublisherError(Exception):
    pass


def _title_and_premise(produced: ProducedEpisode | ProducedDialogueEpisode) -> tuple[str, str]:
    source = produced.episode if isinstance(produced, ProducedEpisode) else produced.screenplay
    return source.title, source.premise


class PublishingAgent:
    def __init__(self, clients: dict[str, PublishingClient]):
        self._clients = clients

    def publish_episode(
        self,
        produced: ProducedEpisode | ProducedDialogueEpisode,
        platforms: list[str],
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        privacy_status: str = "unlisted",
    ) -> list[PublishResult]:
        if produced.stitched_path is None:
            raise PublisherError(
                "This episode has no stitched_path - a multi-scene episode needs a remote_exec stitcher configured in EpisodePipeline before it can be published"
            )

        title, premise = _title_and_premise(produced)
        results: list[PublishResult] = []
        for platform in platforms:
            client = self._clients.get(platform)
            if client is None:
                results.append(PublishResult(platform=platform, success=False, error=f"No client configured for platform '{platform}'"))
                continue
            try:
                video_id = client.upload_video(
                    produced.stitched_path,
                    title=title,
                    description=description or premise,
                    tags=tags or [],
                    privacy_status=privacy_status,
                )
                results.append(PublishResult(platform=platform, success=True, video_id=video_id))
            except Exception as exc:
                results.append(PublishResult(platform=platform, success=False, error=str(exc)))
        return results
