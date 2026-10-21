"""Publishing Agent: the last step - takes a produced dialogue episode's
stitched video and uploads it to whichever configured platforms the
owner names. No reasoning call here (unlike DialogueWriterAgent) -
title/description come straight from the DialogueScreenplay the owner
already approved.

Deliberately per-platform fault-tolerant: one platform's failure (missing
client, expired token, a real upload error) does not stop the others -
every platform gets its own PublishResult, success or not.

Used to also accept the old pipeline's ProducedEpisode (a union type,
with an isinstance check picking `.episode` vs `.screenplay` for the
title/premise) - removed 2026-10-21 along with that whole pipeline
(WriterAgent/DirectorAgent/EpisodePipeline, see models.py's own
docstring) per the owner's standing instruction that no narrator-style
code may exist in this repository at all. Only ProducedDialogueEpisode
is produced by the live bot now, so only it is accepted here.
"""
from __future__ import annotations

from src.integrations.publishing_client import PublishingClient
from src.models import ProducedDialogueEpisode, PublishResult


class PublisherError(Exception):
    pass


class PublishingAgent:
    def __init__(self, clients: dict[str, PublishingClient]):
        self._clients = clients

    def publish_episode(
        self,
        produced: ProducedDialogueEpisode,
        platforms: list[str],
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        privacy_status: str = "unlisted",
    ) -> list[PublishResult]:
        if produced.stitched_path is None:
            raise PublisherError(
                "This episode has no stitched_path - the final merge must succeed before it can be published"
            )

        title, premise = produced.screenplay.title, produced.screenplay.premise
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
