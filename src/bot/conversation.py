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

Owner's instruction, 2026-10-19 ("يجب ان ينفذ كل حرف و كل كلمة كتبتها
بدون حذف او نقصان ... أنا من أدفع المال و أنا حر ماذا أريد" - every
letter and word I wrote must be used, no deletion, no shortening - I'm
the one paying, I'm free to choose what I want): `/new [seconds]` still
picks a line_count from a target duration, which can mean the writer
has to choose which quoted moments to keep over others - acceptable
when the owner wants a short, cheap preview, but not what this
instruction asks for. `/full` is a separate command for exactly that
case: no duration, no estimate, no choosing - it drives
`write_complete_dialogue_screenplay()`, which uses every real quoted
moment in the story, however many that turns out to be, and the owner
pays for exactly that - shown in the same cost estimate before approval,
same as `/new`.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field

from src.agents.dialogue_writer import DialogueWriterAgent, DialogueWriterError, count_quoted_moments
from src.agents.publisher import PublishingAgent
from src.integrations.replicate_image_client import PRICE_PER_IMAGE_USD_UNVERIFIED
from src.integrations.replicate_video_client import PROVISIONAL_PRICE_PER_SECOND_USD
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

# Real gap found, 2026-10-20 ("لماذا النهاية متقطعة في البوت؟ لماذا لا
# يكون كله في ميساج واحد؟" - why is the ending cut off in the bot, why
# isn't it all in one message?): the comment above about a real ~900-word
# story arriving cut off describes Telegram's own CLIENT auto-splitting
# what a human TYPES into the app (the incoming direction, already
# handled by handle_plain_text's accumulation). The Bot API's own
# sendMessage call the BOT uses to reply is different and was never
# covered: it has the same real 4096-character limit, but does not split
# for you - it rejects the whole call outright ("Bad Request: message is
# too long"). A long screenplay or a produced-result message with many
# failed/skipped lines could hit this with no chunking logic anywhere -
# runner.py's _send_reply sent reply.text in one single call. Kept at
# 4000, not 4096, for margin (Telegram counts each newline as 2 of the
# 4096, per real developer reports - not officially documented, so this
# stays conservative rather than cutting it exactly at the edge).
TELEGRAM_MESSAGE_LIMIT = 4000


# Owner's report, 2026-10-24: the bot showed a made-up "$16,000" for
# "wages and crew". No code path writes that - it came from LLM-written
# text. The only cost the owner may ever see is the one computed here in
# code, so any LLM reply that mentions money is replaced, never shown.
COST_LABEL = "تكلفة السيرفر الرقمي لتوليد الصور والفيديو"
_MONEY_PATTERN = re.compile(
    r"[$€£]|\d\s*(?:usd|eur)\b|دولار|يورو|أجور|اجور|طاقم|ميزاني|تكلف|budget|wage|crew|\bcosts?\b",
    re.IGNORECASE,
)


def split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Splits text into chunks the Bot API will actually accept, breaking
    at the last newline before the limit when one exists so a sentence or
    a screenplay line is never cut mid-word - only falling back to a hard
    cut when a single chunk has no newline to break on at all."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


@dataclass
class BotReply:
    text: str
    video_path: str | None = None
    # Real incident, 2026-10-20: when the final merge into one stitched
    # episode fails, the real individual clips (each one already paid for
    # and sitting finished on disk) used to simply never reach the owner -
    # BotReply could only carry a single video. video_paths lets
    # format_produced hand over every real clip it actually has instead of
    # silently keeping them on a server the owner has no access to.
    video_paths: list[str] = field(default_factory=list)
    buttons: list[tuple[str, str]] | None = None


@dataclass
class _ChatState:
    screenplay: DialogueScreenplay | None = None
    produced: ProducedDialogueEpisode | None = None
    pending_text: str = ""


class BotConversation:
    def __init__(
        self,
        dialogue_writer: DialogueWriterAgent,
        publisher: PublishingAgent,
        # Real finding, 2026-10-21: the real rate for whichever provider
        # is actually wired in (src/app.py, same value passed to
        # DialoguePipeline's own price_per_second_usd) must be passed
        # explicitly here - never hard-coded, which is exactly the
        # "estimate doesn't match the real provider" defect this project
        # spent the whole evening fixing for fal.ai's own rate before
        # fal.ai was deleted outright.
        price_per_second_usd: float = PROVISIONAL_PRICE_PER_SECOND_USD,
        # Owner's instruction, 2026-10-21 ("قم بالتنظيف ... كثرة الكودات
        # اصبحت تتداخل في بعضها"): same reasoning as price_per_second_usd
        # above, for the portrait cost component of the estimate.
        price_per_image_usd: float = PRICE_PER_IMAGE_USD_UNVERIFIED,
        # Cinematic shot mode: the estimated cost of one flux-2-dev shot
        # still, added once per spoken line. 0.0 when the mode is off.
        estimated_shot_image_usd: float = 0.0,
    ):
        self._estimated_shot_image_usd = estimated_shot_image_usd
        self._dialogue_writer = dialogue_writer
        self._publisher = publisher
        self._price_per_second_usd = price_per_second_usd
        self._price_per_image_usd = price_per_image_usd
        self._states: dict[str, _ChatState] = {}

    def handle_plain_text(self, chat_id: str, text: str) -> None:
        text = text.strip()
        if not text:
            return None
        state = self._states.setdefault(chat_id, _ChatState())
        state.pending_text = (state.pending_text + "\n" + text).strip() if state.pending_text else text
        return None

    def handle_message(self, chat_id: str, text: str) -> BotReply:
        # Owner's standing instruction, 2026-10-21 ("اريد الحديث معه كما
        # اتحدث معك ... محادثة حرة كاملة مع نقاش وتعديل جماعي للقصة" - I
        # want to talk to it like I talk to you, a full free conversation
        # with discussion and collaborative editing of the story): every
        # plain (non-command) message is routed through
        # DialogueWriterAgent.route_message() into one of a fixed set of
        # real actions - never arbitrary code. The one real-money moment
        # (production) is NEVER one of those actions, under any phrasing
        # - it still only ever starts from the explicit ✅ button shown
        # under a screenplay (handle_approve), completely unchanged by
        # anything below. /new, /full, /edit, /caption keep working
        # exactly as before for anyone who prefers exact commands.
        text = text.strip()
        state = self._states.setdefault(chat_id, _ChatState())
        # Real risk found by code review, 2026-10-21: Telegram's own
        # CLIENT still auto-splits one long typed/pasted message into
        # several separate text updates (see the 2026-10-19 finding
        # above) - runner.py debounces these into one real call here,
        # arriving with an EMPTY text once the owner has gone quiet, so
        # this reads the full coalesced text out of pending_text itself
        # rather than routing a mid-sentence fragment through the LLM.
        # A caller passing a real, non-empty message (e.g. a direct unit
        # test) is unaffected - this path only ever triggers when there
        # is nothing new to add on top of what is already accumulated.
        #
        # Real, live race found 2026-10-21, the SECOND time this file was
        # fixed for pending_text loss that same evening: the first fix
        # (see git history) pulled pending_text into `text` and cleared
        # it to "" immediately, intending to restore it in every branch
        # that does not consume it. That still left a real window open -
        # route_message() below is a genuine, slow, synchronous network
        # call (now run via loop.run_in_executor from runner.py, exactly
        # so it does not freeze the whole bot - see that file's own
        # history) - and anything else touching state.pending_text
        # DURING that window (most concretely: the owner's own very next
        # /full or /new, now also running on its own executor thread)
        # could observe it already cleared, before this function ever got
        # back around to restoring it. The owner hit this directly and
        # repeatedly: send the story, then /full immediately after -
        # "nothing to build" every time, even though the story had just
        # been sent.
        #
        # Fixed by never speculatively clearing state.pending_text at
        # all. It is only ever read here, and only ever mutated at the
        # ONE real point something actually consumes it (a successful
        # screenplay build, below) - so there is no window, of any
        # length, where a concurrent read of it sees anything but its
        # real, true content.
        pulled_from_pending = not text
        source_text = state.pending_text if pulled_from_pending else text
        if not source_text:
            return BotReply(text="")

        try:
            routed = self._dialogue_writer.route_message(
                state.screenplay, "" if pulled_from_pending else state.pending_text, source_text
            )
        except DialogueWriterError as exc:
            return BotReply(text=f"تعذّر فهم رسالتك: {exc}")

        if _MONEY_PATTERN.search(routed.reply):
            routed.reply = self._real_cost_reply(state.screenplay)

        # Owner's instruction, 2026-10-24: "I send an idea only, and the
        # system writes the whole film". With no draft yet, story text is
        # never parked waiting for more - it becomes the screenplay now.
        if routed.intent == "accumulate" and state.screenplay is None:
            routed.intent = "build_screenplay"

        if routed.intent == "accumulate":
            if not pulled_from_pending:
                self.handle_plain_text(chat_id, source_text)
            # Already pulled straight from pending_text itself above - it
            # already holds exactly this content, nothing to add.
            return BotReply(text=routed.reply)

        if routed.intent == "discuss":
            return BotReply(text=routed.reply)

        if routed.intent == "build_screenplay":
            # Real, live bug found 2026-10-21, directly from the owner's
            # own confusion ("لم اعد افهمك... لم اعد افهم البوت"): with
            # handle_message() only ever reachable from runner.py's
            # debounce path, a story that looked "ready to build" used to
            # auto-build and CONSUME pending_text while the owner's own
            # /full or /new could still be racing it on another thread -
            # two real, separate fixes that evening (see docs/SOURCES.md)
            # closed that race structurally: pending_text is never cleared
            # speculatively any more (read above), and runner.py now
            # cancels any still-pending debounce task the instant an
            # explicit command arrives. With the actual race gone, this
            # no longer needs to refuse auto-building outright.
            #
            # Owner's explicit instruction, same evening, once that race
            # was actually fixed ("السيناريو تلقائيا مع الاختيار يبقى هو
            # لي" - build it automatically, but the choice stays mine):
            # free conversation builds the screenplay on its own once the
            # owner goes quiet - no /full or /new needed for the common
            # case - but an explicit /full or /new sent before that still
            # always wins (cancels the pending auto-build outright via
            # runner.py's _cancel_debounce, never just races it), so the
            # owner's own explicit choice of mode is never silently
            # pre-empted by the automatic path.
            # Real bug caught by the test for this exact change, 2026-10-21:
            # when pulled_from_pending, source_text already IS
            # state.pending_text (read verbatim at the top of this
            # method) - concatenating them again here duplicated the
            # owner's own words, which then counted every real quoted
            # moment twice (count_quoted_moments on the doubled text) and
            # asked the model for twice as many lines as the real premise
            # actually has. Only the direct-call, non-empty-text path has
            # genuinely separate pending_text + new text to combine.
            premise = (
                source_text
                if pulled_from_pending
                else (f"{state.pending_text}\n{source_text}".strip() if state.pending_text else source_text)
            )
            if not premise:
                return BotReply(text=routed.reply)
            try:
                quote_count = count_quoted_moments(premise)
            except DialogueWriterError as exc:
                return BotReply(text=str(exc))
            title = premise[:50]
            try:
                # Real quoted moments exist: build verbatim (the same
                # guarantee /full gives) - the owner's own words are
                # never at risk of being selected/invented over, which
                # is exactly what a free conversational trigger must not
                # silently do. Only a genuinely quote-free idea falls
                # back to the estimated/selective mode /new uses, since
                # there is nothing verbatim to preserve either way.
                if quote_count > 0:
                    screenplay = self._dialogue_writer.write_complete_dialogue_screenplay(title, premise)
                else:
                    line_count = max(2, round(DEFAULT_TARGET_SECONDS / ESTIMATED_SECONDS_PER_LINE))
                    screenplay = self._dialogue_writer.write_dialogue_screenplay(title, premise, line_count)
            except DialogueWriterError as exc:
                return BotReply(text=str(exc))
            state.screenplay = screenplay
            state.pending_text = ""
            return BotReply(
                text=f"{routed.reply}\n\n{self._format_screenplay(screenplay)}",
                buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")],
            )

        if routed.intent == "edit_lines":
            # route_message() already refuses this intent when no draft
            # exists (a real code-level gate, not just a prompt rule) -
            # this mirrors handle_edit's own guard only as defense in depth.
            if state.screenplay is None:
                return BotReply(text="لا توجد مسودة سيناريو بعد - صف لي القصة أولًا.")
            not_found = self._apply_edits(state.screenplay, routed.edits)
            reply_text = f"{routed.reply}\n\n{self._format_screenplay(state.screenplay)}"
            if not_found:
                reply_text += f"\n\n⚠️ لم توجد هذه الأرقام في المسودة الحالية (تحتوي على {len(state.screenplay.lines)} سطراً): {', '.join(str(n) for n in not_found)}"
            return BotReply(text=reply_text, buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

        if routed.intent == "set_captions":
            if state.screenplay is None:
                return BotReply(text="لا توجد مسودة سيناريو بعد - صف لي القصة أولًا.")
            not_found = self._apply_captions(state.screenplay, routed.caption_lines)
            reply_text = f"{routed.reply}\n\n{self._format_screenplay(state.screenplay)}"
            if not_found:
                reply_text += f"\n\n⚠️ لم توجد هذه الأرقام في المسودة الحالية (تحتوي على {len(state.screenplay.lines)} سطراً): {', '.join(str(n) for n in not_found)}"
            return BotReply(text=reply_text, buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

        # Unreachable: route_message() validates intent is one of the
        # five handled above and raises DialogueWriterError otherwise.
        # pending_text was never touched above, so nothing to restore.
        return BotReply(text=routed.reply)

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

        # Real incident, 2026-10-21: a real story with 25 real quoted
        # moments was sent via /new (default 60s -> exactly 15 lines,
        # 60/ESTIMATED_SECONDS_PER_LINE), not /full - the owner meant to
        # use every quote verbatim but /new's own LLM prompt is free to
        # SELECT or invent when there are more real quotes than
        # line_count allows, which is exactly what /full exists to
        # prevent. The real $6.838 that bought the wrong result, and a
        # wrong video that reached a real public YouTube URL before
        # anyone caught it, were spent because the bot let this through
        # silently - the two commands are too easy to confuse (same
        # story pasted first, one letter apart in spirit) to rely on the
        # owner remembering correctly under pressure every time. If the
        # real quote count already exceeds what this line_count can
        # hold, this is refused before any real cost, not after.
        try:
            quote_count = count_quoted_moments(premise)
        except DialogueWriterError as exc:
            return BotReply(text=str(exc))
        if quote_count > line_count:
            return BotReply(
                text=(
                    f"⚠️ نصك يحتوي {quote_count} اقتباسًا حقيقيًا (كلامًا منقولًا حرفيًا بين «» أو \"\")، لكن "
                    f"/new بالمدة التي أرسلتها سيبني {line_count} سطرًا فقط - أي أن الذكاء الاصطناعي سيضطر لاختيار "
                    "بعض كلامك الحقيقي ويترك الباقي، أو حتى يخترع ما لم تكتبه.\n\n"
                    "إن كنت تريد استخدام كل كلمة كتبتها حرفيًا بدون اختيار ولا اختراع، أرسل /full بدلاً من /new.\n"
                    f"إن كنت تقصد فعلاً اختيار بعض الاقتباسات فقط، أرسل /new بمدة أطول "
                    f"({int(quote_count * ESTIMATED_SECONDS_PER_LINE)} ثانية على الأقل)."
                )
            )

        # Real incident, 2026-10-22: a real Gemini 500 (after Groq's own
        # fallback to it also failed) reached the owner as a raw,
        # uncaught exception via runner.py's generic top-level error
        # handler - "❌ حدث خطأ حقيقي غير متوقع: INTERNAL. {...}" - not
        # the bot's normal reply path. DialogueWriterAgent now wraps
        # every real reasoning-client call so any failure becomes a real
        # DialogueWriterError (see its own _complete() docstring), but
        # that is only useful if it is actually caught here - /new and
        # /full never had this guard at all, unlike the equivalent
        # free-conversation build_screenplay branch in handle_message()
        # above, which already catches it.
        try:
            screenplay = self._dialogue_writer.write_dialogue_screenplay(title, premise, line_count)
        except DialogueWriterError as exc:
            return BotReply(text=str(exc))
        state.screenplay = screenplay
        state.pending_text = ""
        return BotReply(text=self._format_screenplay(screenplay), buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    def handle_full(self, chat_id: str, args_text: str) -> BotReply:
        args_text = args_text.strip()
        state = self._states.setdefault(chat_id, _ChatState())

        # Owner's instruction, 2026-10-19: no duration, no estimate, no
        # choosing which quoted moments to keep - every one of them, in
        # order, verbatim, however many lines that turns out to be.
        premise = f"{state.pending_text}\n{args_text}".strip() if state.pending_text else args_text

        if not premise:
            return BotReply(text="الاستخدام: أرسل قصتك كاملة كرسالة عادية (يمكن أن تكون طويلة وتنقسم لعدة رسائل)، ثم أرسل /full لبناء السيناريو من كل كلمة فيها بدون حذف.")

        title = premise[:50]
        # Real incident, 2026-10-22: see handle_new()'s own comment above
        # - /full had the identical gap.
        try:
            screenplay = self._dialogue_writer.write_complete_dialogue_screenplay(title, premise)
        except DialogueWriterError as exc:
            return BotReply(text=str(exc))
        state.screenplay = screenplay
        state.pending_text = ""
        return BotReply(text=self._format_screenplay(screenplay), buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    def handle_edit(self, chat_id: str, args_text: str) -> BotReply:
        # Owner's instruction, 2026-10-20 ("لماذا كتبت يعدل مشهد واحد؟ بل
        # يعدل كل المشاهد فالحرية لي" - why did you write it edits one
        # scene? it should edit ALL scenes, the freedom is mine): /edit
        # used to accept exactly one "<number> <text>" pair per command.
        # It now accepts several, one per real line of the message (runner.py
        # passes the raw, newline-preserving message text here instead of
        # context.args, which collapses newlines into spaces and would
        # make several edits unparseable) - every edit in the message is
        # applied in one pass, and the owner gets the updated screenplay
        # back once, not once per line.
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="لا توجد مسودة سيناريو بعد - ابدأ بـ /new.")

        edits: list[tuple[int, str]] = []
        for raw_line in args_text.strip().split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parts = raw_line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                edits.append((int(parts[0]), parts[1]))

        if not edits:
            return BotReply(
                text=(
                    "الاستخدام: /edit <رقم السطر> <الحوار الجديد>\n"
                    "يمكنك تعديل عدة أسطر دفعة واحدة - كل تعديل في سطر منفصل من نفس الرسالة، مثلًا:\n"
                    "/edit\n1 حوار جديد للسطر الأول\n3 حوار جديد للسطر الثالث"
                )
            )

        not_found = self._apply_edits(state.screenplay, edits)
        reply_text = self._format_screenplay(state.screenplay)
        if not_found:
            reply_text += f"\n\n⚠️ لم توجد هذه الأرقام في المسودة الحالية (تحتوي على {len(state.screenplay.lines)} سطراً): {', '.join(str(n) for n in not_found)}"
        return BotReply(text=reply_text, buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    @staticmethod
    def _apply_edits(screenplay: DialogueScreenplay, edits: list[tuple[int, str]]) -> list[int]:
        """Applies each (1-based line number, new text) edit, returns the
        numbers that did not match any real line. Shared by /edit and the
        conversational router (handle_message) - one real implementation,
        not two that could silently drift apart."""
        not_found: list[int] = []
        for line_number, new_text in edits:
            matching = [l for l in screenplay.lines if l.index == line_number - 1]
            if not matching:
                not_found.append(line_number)
                continue
            matching[0].line = new_text
        return not_found

    @staticmethod
    def _apply_captions(screenplay: DialogueScreenplay, numbers: list[int]) -> list[int]:
        """Marks each 1-based line number spoken=False, returns the numbers
        that did not match any real line. Shared by /caption and the
        conversational router (handle_message)."""
        not_found: list[int] = []
        for number in numbers:
            matching = [l for l in screenplay.lines if l.index == number - 1]
            if not matching:
                not_found.append(number)
                continue
            matching[0].spoken = False
        return not_found

    def handle_caption(self, chat_id: str, args_text: str) -> BotReply:
        # Owner's instruction, 2026-10-20 ("نحذف الام و يبقى حوارها مكتوب
        # على الشاشة ... لكي نستحقظ على المال" - remove the mother as a
        # speaking role, her dialogue stays written on screen, to save
        # money): marks specific line numbers as on-screen text instead
        # of spoken/lip-synced - real zero cost for those lines
        # (DialoguePipeline.produce() skips TTS and Replicate entirely
        # for any line with spoken=False). Same multi-number-per-message
        # pattern as /edit, reusing the exact same freedom the owner
        # insisted on there.
        state = self._states.get(chat_id)
        if state is None or state.screenplay is None:
            return BotReply(text="لا توجد مسودة سيناريو بعد - ابدأ بـ /new.")

        numbers: list[int] = []
        for token in args_text.split():
            if token.isdigit():
                numbers.append(int(token))

        if not numbers:
            return BotReply(
                text=(
                    "الاستخدام: /caption <رقم السطر> [أرقام أخرى] - يجعل هذه الأسطر نصًا مكتوبًا "
                    "على الشاشة بدل صوت ومقطع فيديو حقيقي، بتكلفة حقيقية صفر لها. مثلًا:\n"
                    "/caption 3 7"
                )
            )

        not_found = self._apply_captions(state.screenplay, numbers)
        reply_text = self._format_screenplay(state.screenplay)
        if not_found:
            reply_text += f"\n\n⚠️ لم توجد هذه الأرقام في المسودة الحالية (تحتوي على {len(state.screenplay.lines)} سطراً): {', '.join(str(n) for n in not_found)}"
        return BotReply(text=reply_text, buttons=[("✅ موافقة", "approve"), ("❌ إلغاء", "cancel")])

    def handle_cancel(self, chat_id: str) -> BotReply:
        self._states[chat_id] = _ChatState()
        return BotReply(text="تم إلغاء المسودة.")

    def get_current_screenplay(self, chat_id: str) -> DialogueScreenplay | None:
        """Owner's instruction, 2026-10-21 ("الصور ضعها انت الان و هكذا
        لا ندفع شيء"): the one read-only accessor runner.py's /photo
        handler needs to validate a manually-uploaded character photo
        against the real current screenplay (does this character name
        actually exist in it?) before saving it - _states stays private,
        this is the narrow read the feature actually needs."""
        state = self._states.get(chat_id)
        return state.screenplay if state is not None else None

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
        failed_note = self._format_failed_lines(produced) if produced.failed_lines else ""
        # Owner's instruction, 2026-10-20: the hard spend ceiling
        # (Settings.max_episode_cost_usd / DialoguePipeline.max_cost_usd)
        # stops attempting further lines once reached - the owner must be
        # told exactly which lines were never attempted (and so never
        # billed a cent) rather than finding a shorter result unexplained.
        skipped_note = (
            f"\n\n🛑 توقف التوليد عند الحد الأقصى للتكلفة قبل محاولة {len(produced.skipped_lines)} سطر "
            f"(لم تُصرف أي أموال عليها): {', '.join(str(l.index + 1) for l in produced.skipped_lines)}"
            if produced.skipped_lines
            else ""
        )
        # Real bug found by code review, 2026-10-21: this message still
        # said "(a placeholder photo was used instead)" after the owner's
        # standing instruction to delete that fallback outright - with no
        # fallback left, a portrait failure now means that character's
        # lines FAILED (zero cost, reported in failed_note above), not
        # that production continued with a stand-in face. Left unfixed
        # when the fallback itself was deleted, this message would have
        # told the owner something false about their own real production
        # run - exactly the kind of mismatch they have been catching all
        # session.
        portrait_note = (
            "\n\n🖼️ تعذّر توليد صورة حقيقية لهذه الشخصيات، فلم تُنتَج أسطرها (بدون أي تكلفة - لا توجد صورة احتياطية بأمر المالك):\n"
            + "\n".join(f"- {name}: {error}" for name, error in produced.portrait_failures)
            if produced.portrait_failures
            else ""
        )
        if produced.rejected_cost_usd:
            portrait_note += f"\n\n🔁 لقطات أعدتَها (مدفوعة، ليست في الفيديو): ${produced.rejected_cost_usd:.3f} - محسوبة في المجموع."
        if produced.stitched_path is None:
            # Real incident, 2026-10-20: a real $8.18 run's clips were all
            # genuinely finished, but the merge step failed and the owner
            # received NEITHER the stitched video NOR the individual real
            # clips it already paid for - "تعذّر دمجها" with no video
            # attached at all. When there is at least one real finished
            # clip, every one of them is sent individually (video_paths)
            # instead of only ever being described in text, and the real
            # merge failure reason (produced.stitch_error) is shown so a
            # repeat is diagnosable rather than just "تعذّر دمجها" again.
            stitch_note = f"\n\nسبب فشل الدمج: {produced.stitch_error}" if produced.stitch_error else ""
            if produced.clips:
                return BotReply(
                    text=(
                        f"⚠️ تم توليد {len(produced.clips)} مقطع حقيقي بنجاح ({COST_LABEL}، الحقيقية: "
                        f"${produced.total_cost_usd:.3f}) لكن تعذّر دمجها في فيديو واحد - "
                        f"كل مقطع مُرسل أدناه منفصلًا.{stitch_note}{failed_note}{skipped_note}{portrait_note}"
                    ),
                    video_paths=[c.local_path for c in produced.clips],
                )
            return BotReply(
                text=(
                    f"تم توليد 0 مقطع. {COST_LABEL}، الحقيقية حتى الآن: ${produced.total_cost_usd:.3f}"
                    f"{stitch_note}{failed_note}{skipped_note}{portrait_note}"
                )
            )
        return BotReply(
            text=(
                f"✅ تم. {len(produced.clips)} مقطع حوار، {COST_LABEL}، الحقيقية: "
                f"${produced.total_cost_usd:.3f}.{failed_note}{skipped_note}{portrait_note}"
            ),
            video_path=produced.stitched_path,
            buttons=[("📤 نشر على YouTube", "publish:youtube"), ("🗑️ تجاهل", "discard")],
        )

    @staticmethod
    def _format_failed_lines(produced: ProducedDialogueEpisode) -> str:
        # Real finding, 2026-10-19: a real line timing out on fal.ai used
        # to discard every clip already paid for; now the ones that
        # failed are reported by number so the owner knows exactly which
        # real money bought nothing and which lines to retry with /edit.
        #
        # Real bug found by code review, same day: this used to claim
        # EVERY failed line's cost "was not counted" - false whenever
        # the real fal.ai call already succeeded (and was billed) and
        # only a later local step failed. DialoguePipeline now prefixes
        # exactly those messages with a real billing note, so they are
        # shown per line instead of one blanket (sometimes false) claim.
        details = "\n".join(f"- سطر {l.index + 1}: {error}" for l, error in produced.failed_lines)
        return f"\n\n⚠️ فشلت {len(produced.failed_lines)} أسطر - يمكنك تعديلها وإعادة المحاولة:\n{details}"

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

    def _estimated_cost(self, screenplay: DialogueScreenplay) -> float:
        spoken_count = sum(1 for l in screenplay.lines if l.spoken)
        # Real bug found by code review, 2026-10-21: this estimate only
        # ever counted spoken-line video cost - never the real portrait
        # generation cost DialoguePipeline also spends, once per character
        # who appears in any line at all (spoken or captioned).
        appearing_character_count = len({l.character for l in screenplay.lines})
        # Uses self._price_per_second_usd - the real rate for whichever
        # provider is actually wired in (src/app.py), never a hard-coded
        # fal.ai-specific constant. See __init__'s own comment above.
        return (
            spoken_count * ESTIMATED_SECONDS_PER_LINE * self._price_per_second_usd
            + appearing_character_count * self._price_per_image_usd
            + spoken_count * self._estimated_shot_image_usd
        )

    def _real_cost_reply(self, screenplay: DialogueScreenplay | None) -> str:
        if screenplay is None:
            return f"{COST_LABEL} يحسبها النظام فقط، وتظهر تحت السيناريو بعد كتابته."
        return (
            f"{COST_LABEL} (يحسبها النظام فقط، تقديرية): ${self._estimated_cost(screenplay):.2f}. "
            f"القيمة الحقيقية تظهر بعد التوليد الفعلي."
        )

    def _format_screenplay(self, screenplay: DialogueScreenplay) -> str:
        lines = [f"📝 {screenplay.title}", "", "الشخصيات:"]
        for c in screenplay.characters:
            lines.append(f"- {c.name}")
        lines.append("")
        # Owner's own definition of "فيلم سينمائي", 2026-10-23 ("مشاهد
        # متعددة" - multiple scenes): the owner reads/edits this exact
        # text before any real money is spent (the project's one human
        # gate) - a real scene break that only exists inside
        # screenplay.scenes/DialogueLine.scene and never reaches this
        # message would be invisible to the one person who is supposed to
        # review it. `scenes_by_index` stays empty for any screenplay
        # written before this existed (models.py's own additive-field
        # guarantee), so the loop below falls back to the flat,
        # unnumbered rendering every existing test/fixture already
        # expects - never a behavior change for a screenplay with no real
        # scene structure.
        scenes_by_index = {s.index: s for s in screenplay.scenes}
        current_scene_index: int | None = None
        for l in screenplay.lines:
            if scenes_by_index and l.scene != current_scene_index:
                current_scene_index = l.scene
                scene = scenes_by_index.get(current_scene_index)
                if scene is not None:
                    lines.append(f"🎬 المشهد {current_scene_index + 1}: {scene.location} ({scene.mood})")
            # Owner's instruction, 2026-10-20: a line marked /caption must
            # be visibly distinct here BEFORE approval - otherwise the
            # owner has no way to confirm which lines will actually cost
            # money and which will be free on-screen text.
            marker = " (نص على الشاشة - بدون صوت، تكلفة $0)" if not l.spoken else ""
            lines.append(f"{l.index + 1}. {l.character}: {l.line}{marker}")
        lines.append("")
        est_cost = self._estimated_cost(screenplay)
        lines.append(f"({COST_LABEL}، تقديرية: ${est_cost:.2f} - القيمة الحقيقية تُحسب بعد التوليد الفعلي)")
        lines.append("لتعديل سطر أو عدة أسطر دفعة واحدة، كل تعديل في سطر منفصل:\n/edit\n<الرقم> <الحوار الجديد>\n<رقم آخر> <حوار آخر>")
        lines.append("لجعل سطر (أو عدة أسطر) نصًا مكتوبًا على الشاشة بدل صوت حقيقي (بتكلفة $0): /caption <الرقم> [أرقام أخرى]")
        return "\n".join(lines)
