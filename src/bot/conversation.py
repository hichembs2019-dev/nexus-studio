"""The Telegram bot's actual decision logic, kept entirely synchronous and
free of the telegram library itself - testable with Mock agents and no
live bot. The thin async wiring that turns real Telegram updates into
calls on this class (and runs the slow DialoguePipeline.produce() call in
a background thread so the bot stays responsive) lives in
src/bot/runner.py, the same "thinnest possible wrapper, not unit tested"
split nexus-commerce's own telegram_listener.py documents and uses.

One chat (the owner's) is assumed to work on one draft at a time - no
concurrency within a chat_id.

Owner's instruction, 2026-10-19 ("بل اطلب من البوت مثلما كان يفعل يرسل
لي السيناريو و يسالني الموافقة" - have the bot do what it used to: send
me the screenplay and ask for approval): this is the dialogue-pipeline
rewrite of what the EpisodePipeline/WriterAgent/DirectorAgent version of
this file did - same approve/cancel human gate before anything real
generates, now driving DialogueWriterAgent + DialoguePipeline (real
character dialogue with lip sync) instead of the old silent-T2V-plus-
narration pipeline, which the live bot was still running right up to
this change despite nothing ever having called it live (confirmed: a
real story sent to the bot got no reply at all, because nothing is
deployed to run either pipeline anywhere yet - see README's deployment
section).

Real finding, 2026-10-19 ("القصة يجب ان تكون كما كتبتها" - the story
must be exactly as I wrote it, no shortening): a real ~900-word owner
story sent as a single /new command arrived at the bot cut off mid-word
("...لإحضار اب" instead of "...لإحضار ابني") - Telegram's own client
auto-splits an outgoing text over roughly 4096 characters into several
separate messages, and only the first one carried the /new command; the
rest arrived as ordinary text messages the bot had no handler for and
silently dropped. `handle_plain_text` now accumulates any ordinary
(non-command) message per chat into `_ChatState.pending_text`, in the
order it arrives; `/new` with no premise of its own uses exactly that
accumulated text, unmodified - never summarized, never trimmed - so a
long real story reaches DialogueWriterAgent complete regardless of how
many pieces Telegram split it into.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.dialogue_writer import DialogueWriterAgent
from src.agents.publisher import PublishingAgent
from src.integrations.fal_video_client import PRICE_PER_SECOND_USD
from src.models import DialogueScreenplay, ProducedDialogueEpisode

DEFAULT_TARGET_SECONDS = 60.0

# A real line's spoken length depends on how much text the writer puts in
# it (DialoguePipeline's own docstring: clip length is NOT fixed, it is
# however long that line's TTS audio runs) - this is this project's own
# documented, stated ESTIMATE for turning a target duration into a line
# count before anything is written, not a guaranteed real duration. The
# real total cost is only known after real generation (format_produced
# reports it) - this estimate exists only so /new <seconds> <idea> can
# pick a reasonable line_count up front.
ESTIMATED_SECONDS_PER_LINE = 4.0


@dataclass
class BotReply:
    text: str
    video_path: str | None = None
    buttons: list[tuple[str, str]] | None = None


@dataclass
class _ChatState:
    screenplay: DialogueScreenplay | None = None
    produced: ProducedDialogueEpisode | None = None
    pending_text: str = ""


class BotConversation:
    def __init__(self, dialogue_writer: DialogueWriterAgent, publisher: PublishingAgent):
        self._dialogue_writer = dialogue_writer
        self._publisher = publisher
        self._states: dict[str, _ChatState] = {}

    def handle_plain_text(self, chat_id: str, text: str) -> None:
        text = text.strip()
        if not text:
            return None
        state = self._states.setdefault(chat_id, _ChatState())
        state.pending_text = (state.pending_text + "\n" + text).strip() if state.pending_text else text
        return None

    def handle_new(self, chat_id: str, args_text: str) -> BotReply:
        args_text = args_text.strip()
        state = self._states.setdefault(chat_id, _ChatState())

        target_seconds = DEFAULT_TARGET_SECONDS
        typed_premise = args_text
        if args_text:
            parts = args_text.split(maxsplit=1)
            if parts[0].isdigit():
                target_seconds = float(parts[0])
                typed_premise = parts[1] if len(parts) == 2 else ""

        # The owner's own accumulated text (handle_plain_text, possibly
        # several Telegram-split messages) always comes first and is
        # never shortened - it IS the real story, verbatim.
        premise = f"{state.pending_text}\n{typed_premise}".strip() if state.pending_text else typed_premise

        if not premise:
            return BotReply(text="الاستخدام: أرسل قصتك كرسالة عادية (يمكن أن تكون طويلة وتنقسم لعدة رسائل)، ثم أرسل /new [عدد الثواني] لبناء السيناريو منها.\nأو مباشرة: /new 30 رائد فضاء وحيد يجد إشارة على المريخ.")

        title = premise[:50]
        line_count = max(2, round(target_seconds / ESTIMATED_SECONDS_PER_LINE))
        screenplay = self._dialogue_writer.write_dialogue_screenplay(title, premise, line_count)
        state.screenplay = screenplay
        state.pending_text = ""
        return BotReply(text=self._format_screenplay(screenplay), buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    def handle_edit(self, chat_id: str, args_text: str) -> BotReply:
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="لا توجد مسودة سيناريو بعد - ابدأ بـ /new.")

        parts = args_text.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            return BotReply(text="الاستخدام: /edit <رقم السطر> <الحوار الجديد>")

        line_number = int(parts[0])
        new_text = parts[1]
        matching = [l for l in state.screenplay.lines if l.index == line_number - 1]
        if not matching:
            return BotReply(text=f"لا يوجد سطر رقم {line_number} في المسودة الحالية (تحتوي على {len(state.screenplay.lines)} سطراً).")

        matching[0].line = new_text
        return BotReply(text=self._format_screenplay(state.screenplay), buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    def handle_cancel(self, chat_id: str) -> BotReply:
        self._states[chat_id] = _ChatState()
        return BotReply(text="تم إلغاء المسودة.")

    def handle_approve(self, chat_id: str) -> tuple[BotReply, DialogueScreenplay | None]:
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="لا توجد مسودة سيناريو للموافقة عليها - ابدأ بـ /new."), None

        screenplay = state.screenplay
        reply = BotReply(
            text=(
                f"⏳ جارٍ توليد {len(screenplay.lines)} مقطع حوار حقيقي بتزامن شفاه فعلي - "
                "يستغرق هذا دقائق. التكلفة الحقيقية تُحسب من الصوت الفعلي بعد الانتهاء، لا قبله."
            )
        )
        return reply, screenplay

    def format_produced(self, chat_id: str, produced: ProducedDialogueEpisode) -> BotReply:
        self._states.setdefault(chat_id, _ChatState()).produced = produced
        if produced.stitched_path is None:
            return BotReply(
                text=(
                    f"تم توليد {len(produced.clips)} مقطع لكن تعذّر دمجها. "
                    f"التكلفة الحقيقية حتى الآن: ${produced.total_cost_usd:.3f}"
                )
            )
        return BotReply(
            text=f"✅ تم. {len(produced.clips)} مقطع حوار، التكلفة الحقيقية ${produced.total_cost_usd:.3f}.",
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
    def _format_screenplay(screenplay: DialogueScreenplay) -> str:
        lines = [f"📝 {screenplay.title}", "", "الشخصيات:"]
        for c in screenplay.characters:
            lines.append(f"- {c.name}")
        lines.append("")
        for l in screenplay.lines:
            lines.append(f"{l.index + 1}. {l.character}: {l.line}")
        lines.append("")
        est_cost = len(screenplay.lines) * ESTIMATED_SECONDS_PER_LINE * PRICE_PER_SECOND_USD["480p"]
        lines.append(f"(تكلفة تقديرية تقريبية: ${est_cost:.2f} - التكلفة الحقيقية تُحسب بعد التوليد الفعلي)")
        lines.append("لتعديل سطر: /edit <الرقم> <الحوار الجديد>")
        return "\n".join(lines)
