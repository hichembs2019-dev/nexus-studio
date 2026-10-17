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
            return BotReply(text="الاستخدام: /new [عدد الثواني] <الفكرة>\nمثال: /new 30 رائد فضاء وحيد يجد إشارة على المريخ.")

        target_seconds = DEFAULT_TARGET_SECONDS
        premise = args_text
        parts = args_text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            target_seconds = float(parts[0])
            premise = parts[1]

        if not premise:
            return BotReply(text="أعطني فكرة بعد عدد الثواني، مثلاً: /new 30 رائد فضاء وحيد يجد إشارة على المريخ.")

        title = premise[:50]
        screenplay = self._writer.write_screenplay(title, premise, target_seconds)
        self._states[chat_id] = _ChatState(screenplay=screenplay)
        return BotReply(text=self._format_screenplay(screenplay), buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    def handle_edit(self, chat_id: str, args_text: str) -> BotReply:
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="لا توجد مسودة سيناريو بعد - ابدأ بـ /new.")

        parts = args_text.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            return BotReply(text="الاستخدام: /edit <رقم المشهد> <النص الجديد>")

        scene_number = int(parts[0])
        new_text = parts[1]
        matching = [s for s in state.screenplay.scenes if s.index == scene_number - 1]
        if not matching:
            return BotReply(text=f"لا يوجد مشهد رقم {scene_number} في المسودة الحالية (تحتوي على {len(state.screenplay.scenes)} مشهداً).")

        matching[0].text = new_text
        return BotReply(text=self._format_screenplay(state.screenplay), buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    def handle_cancel(self, chat_id: str) -> BotReply:
        self._states[chat_id] = _ChatState()
        return BotReply(text="تم إلغاء المسودة.")

    def handle_approve(self, chat_id: str) -> tuple[BotReply, Episode | None, list[str] | None]:
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="لا توجد مسودة سيناريو للموافقة عليها - ابدأ بـ /new."), None, None

        episode = self._director.plan_episode_from_screenplay(state.screenplay)
        state.episode = episode
        narration_texts = [s.text for s in state.screenplay.scenes]
        reply = BotReply(text=f"⏳ جارٍ توليد {len(episode.scenes)} مشهد على معالج GPU حقيقي - يستغرق هذا بضع دقائق لكل مشهد.")
        return reply, episode, narration_texts

    def format_produced(self, chat_id: str, produced: ProducedEpisode) -> BotReply:
        self._states.setdefault(chat_id, _ChatState()).produced = produced
        if produced.stitched_path is None:
            return BotReply(
                text=(
                    f"تم توليد {len(produced.clips)} مقطع لكن تعذّر دمجها "
                    f"(لا يوجد دامج عن بُعد مُهيَّأ). التكلفة الحقيقية حتى الآن: ${produced.total_cost_usd:.3f}"
                )
            )
        return BotReply(
            text=f"✅ تم. {len(produced.clips)} مشهد، التكلفة الحقيقية ${produced.total_cost_usd:.3f}.",
            video_path=produced.stitched_path,
            buttons=[("📤 نشر على YouTube", "publish:youtube"), ("🗑️ تجاهل", "discard")],
        )

    def handle_publish(self, chat_id: str, platform: str) -> BotReply:
        state = self._states.get(chat_id)
        if state is None or state.produced is None:
            return BotReply(text="لا يوجد شيء جاهز للنشر بعد.")

        results = self._publisher.publish_episode(state.produced, [platform])
        result = results[0]
        if not result.success:
            return BotReply(text=f"❌ فشل النشر على {platform}: {result.error}")
        if platform == "youtube":
            return BotReply(text=f"✅ نُشر على قناة Nexora: https://youtu.be/{result.video_id}")
        return BotReply(text=f"✅ نُشر على {platform}: {result.video_id}")

    @staticmethod
    def _format_screenplay(screenplay: Screenplay) -> str:
        lines = [f"📝 {screenplay.title}", ""]
        for s in screenplay.scenes:
            lines.append(f"{s.index + 1}. {s.text}")
        lines.append("")
        lines.append("لتعديل مشهد: /edit <الرقم> <النص الجديد>")
        return "\n".join(lines)
