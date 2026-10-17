"""The Telegram bot's actual decision logic, kept entirely synchronous and
free of the telegram library itself - testable with Mock agents and no
live bot. The thin async wiring that turns real Telegram updates into
calls on this class (and runs the slow EpisodePipeline.produce() call in
a background thread so the bot stays responsive) lives in
src/bot/runner.py, the same "thinnest possible wrapper, not unit tested"
split nexus-commerce's own telegram_listener.py documents and uses.

One chat (the owner's) is assumed to work on one draft at a time - no
concurrency within a chat_id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.director import DirectorAgent
from src.agents.publisher import PublishingAgent
from src.agents.writer import WriterAgent
from src.models import Episode, ProducedEpisode, Screenplay

DEFAULT_TARGET_SECONDS = 30.0


@dataclass
class BotReply:
    text: str
    video_path: str | None = None
    buttons: list[tuple[str, str]] | None = None


@dataclass
class _ChatState:
    screenplay: Screenplay | None = None
    episode: Episode | None = None
    produced: ProducedEpisode | None = None


class BotConversation:
    def __init__(self, writer: WriterAgent, director: DirectorAgent, publisher: PublishingAgent):
        self._writer = writer
        self._director = director
        self._publisher = publisher
        self._states: dict[str, _ChatState] = {}

    def handle_new(self, chat_id: str, args_text: str) -> BotReply:
        args_text = args_text.strip()
        if not args_text:
            return BotReply(text="Usage: /new [seconds] <idea>\nExample: /new 30 A lone astronaut finds a signal on Mars.")

        target_seconds = DEFAULT_TARGET_SECONDS
        premise = args_text
        parts = args_text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            target_seconds = float(parts[0])
            premise = parts[1]

        if not premise:
            return BotReply(text="Give me an idea after the number of seconds, e.g. /new 30 A lone astronaut finds a signal on Mars.")

        title = premise[:50]
        screenplay = self._writer.write_screenplay(title, premise, target_seconds)
        self._states[chat_id] = _ChatState(screenplay=screenplay)
        return BotReply(text=self._format_screenplay(screenplay), buttons=[("✅ Approve", "approve"), ("❌ Cancel", "cancel")])

    def handle_edit(self, chat_id: str, args_text: str) -> BotReply:
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="No draft screenplay yet - start with /new.")

        parts = args_text.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            return BotReply(text="Usage: /edit <scene number> <new text>")

        scene_number = int(parts[0])
        new_text = parts[1]
        matching = [s for s in state.screenplay.scenes if s.index == scene_number - 1]
        if not matching:
            return BotReply(text=f"No scene {scene_number} in the current draft (it has {len(state.screenplay.scenes)}).")

        matching[0].text = new_text
        return BotReply(text=self._format_screenplay(state.screenplay), buttons=[("✅ Approve", "approve"), ("❌ Cancel", "cancel")])

    def handle_cancel(self, chat_id: str) -> BotReply:
        self._states[chat_id] = _ChatState()
        return BotReply(text="Draft discarded.")

    def handle_approve(self, chat_id: str) -> tuple[BotReply, Episode | None]:
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="No draft screenplay to approve - start with /new."), None

        episode = self._director.plan_episode_from_screenplay(state.screenplay)
        state.episode = episode
        reply = BotReply(text=f"⏳ Generating {len(episode.scenes)} scene(s) on the real GPU - this takes a few minutes per scene.")
        return reply, episode

    def format_produced(self, chat_id: str, produced: ProducedEpisode) -> BotReply:
        self._states.setdefault(chat_id, _ChatState()).produced = produced
        if produced.stitched_path is None:
            return BotReply(
                text=(
                    f"Generated {len(produced.clips)} clip(s) but could not stitch them "
                    f"(no remote stitcher configured). Real cost so far: ${produced.total_cost_usd:.3f}"
                )
            )
        return BotReply(
            text=f"✅ Done. {len(produced.clips)} scene(s), real cost ${produced.total_cost_usd:.3f}.",
            video_path=produced.stitched_path,
            buttons=[("📤 Publish to YouTube", "publish:youtube"), ("🗑️ Discard", "discard")],
        )

    def handle_publish(self, chat_id: str, platform: str) -> BotReply:
        state = self._states.get(chat_id)
        if state is None or state.produced is None:
            return BotReply(text="Nothing produced yet to publish.")

        results = self._publisher.publish_episode(state.produced, [platform])
        result = results[0]
        if not result.success:
            return BotReply(text=f"❌ Publish to {platform} failed: {result.error}")
        if platform == "youtube":
            return BotReply(text=f"✅ Published to Nexora: https://youtu.be/{result.video_id}")
        return BotReply(text=f"✅ Published to {platform}: {result.video_id}")

    @staticmethod
    def _format_screenplay(screenplay: Screenplay) -> str:
        lines = [f"📝 {screenplay.title}", ""]
        for s in screenplay.scenes:
            lines.append(f"{s.index + 1}. {s.text}")
        lines.append("")
        lines.append("Edit a scene: /edit <number> <new text>")
        return "\n".join(lines)
