"""Thin, real-integration-only wiring between python-telegram-bot and
BotConversation - the same split nexus-commerce's own
src/integrations/telegram_listener.py documents and uses: the decision
logic (conversation.py) is pure and unit-tested, this file needs a real
bot token and a live Telegram connection to exercise at all, so it is
kept as small as possible.

CommandHandler's constructor and context.args shape are verified from
python-telegram-bot's own docs (2026-10-17). send_video/reply_video are
not independently doc-verified beyond send_message/InlineKeyboardMarkup,
reused from nexus-commerce's verified usage - confirm on first live run.

Owner's standing instruction, 2026-10-21 ("ما هو 101؟" - pointing at an
earlier audit's line-count table - "و لا تصر على شيء لا اريده" - don't
insist on something I don't want): this file used to also have a sibling,
src/integrations/telegram_client.py - a TelegramClient interface/Real/Mock
wrapper around python-telegram-bot, built early on but never actually
wired in here (this file calls context.bot.* and python-telegram-bot's
Application directly). 101 lines of genuinely unused code, confirmed by
grep before deleting - removed outright, along with its own test file.

DialoguePipeline.produce() is slow (minutes, real per-line TTS + fal.ai
calls) and synchronous; running it directly inside an async handler
would block the whole bot. It runs in the default executor (a
background thread) instead, so /new, /edit and button presses on OTHER
chats keep working while a generation is in flight.

Owner's instruction, 2026-10-19 ("بل اطلب من البوت مثلما كان يفعل يرسل
لي السيناريو و يسالني الموافقة"): this now drives DialoguePipeline
(real character dialogue, lip-synced) instead of the old EpisodePipeline
- the produce() call here is the one real-money moment (fal.ai speech-
to-video spend), gated behind the SAME approve button the old pipeline
used, never called on its own.

Real bug found live, 2026-10-19 ("لماذا كل هذا ... ألم اقل لك بدون
تخمين و لا افتراضيات؟"): the owner pressed Approve and `produce()`
raised immediately (a real Docker-image bug, fixed the same commit in
`Dockerfile`) - but `_produce_and_notify` had no error handling at all,
so `asyncio.create_task(...)`'s exception just died as "Task exception
was never retrieved" in the container's own logs, completely invisible
from Telegram. The owner waited over two hours for a reply that could
never have come - no crash, no error message, nothing. `_produce_and_notify`
now catches any real exception from `produce()` and sends it to the
owner as an actual Telegram message instead of only the container logs -
this is the one place in the whole bot where a failure must reach the
owner's phone directly, since it is the one call nothing else here
retries or falls back from.

Real risk found on review, 2026-10-19: Python's own asyncio docs warn
that the event loop holds only a WEAK reference to a Task created by
asyncio.create_task() - "a task that isn't referenced elsewhere may get
garbage collected at any time, even before it's done." _produce_and_notify
runs for minutes and spends real money; a task silently disappearing
mid-run would look exactly like the two-hour silent-hang incident this
same file already fixed once, from an entirely different cause. Every
create_task() call below now keeps a strong reference in
_background_tasks (module-level per build_application call) until it
finishes, per the exact pattern Python's own docs recommend.

Real gap found by a dedicated code review, 2026-10-19 ("لن اكف عنك الا
ان تراجع كل شيء و تكتشف الاخطاء" - I won't stop until you review
everything and find the bugs): the Approve-button fix above only
covers the one real-money call. `_new`/`_full`/`_edit` call
BotConversation.handle_new()/handle_full()/handle_edit() with no error
handling at all - if DialogueWriterAgent's reasoning call fails (Groq
AND its Gemini fallback both failing, or a malformed reply no retry
recovers from), the exception propagates out of the handler with
nothing sent to Telegram - the exact same "owner gets total silence"
failure class the Approve-button fix exists to prevent, just reachable
from a different command. python-telegram-bot's own Application
supports a single global error_handler for exactly this
(add_error_handler) instead of wrapping every individual handler -
registered below, it reports any uncaught exception from ANY handler
to the chat it happened in.

Owner's standing instruction, 2026-10-21 ("اريد الحديث معه كما اتحدث
معك" - I want to talk to it like I talk to you): plain text no longer
only silently accumulates (BotConversation.handle_plain_text) - it also
now drives a real LLM router (BotConversation.handle_message ->
DialogueWriterAgent.route_message) that lets the owner discuss and
collaboratively edit the screenplay in free Arabic text, not just the
fixed /new, /full, /edit <n> <text>, /caption <n> commands (which still
work unchanged, for anyone who prefers them).

Telegram's own CLIENT still auto-splits one long typed/pasted message
into several separate text updates (the 2026-10-19 finding above) -
routing each fragment through the LLM individually would ask it to
interpret a sentence cut off mid-word. _TEXT_DEBOUNCE_SECONDS below
coalesces rapid-fire fragments per chat: each new fragment cancels any
still-pending timer for that chat and restarts it, so the router only
actually runs once the owner has gone quiet for that long. The pending
asyncio.Task is kept in _debounce_tasks until it fires or is cancelled -
the same strong-reference pattern _background_tasks/_spawn above exists
for, and for the identical documented reason (the event loop holds only
a weak reference to a bare Task).

The one real-money call (dialogue_pipeline.produce(), inside
_produce_and_notify) is untouched by any of this and stays reachable
from exactly one place: the "✅ موافقة" button's own branch in _button
below. BotConversation.handle_message() has no reference to
DialoguePipeline at all - confirmed by grep before writing this, the
same way the dead telegram_client.py file above was confirmed unused
before deleting it - so production cannot be reached from a chat
message however it is phrased, free conversation included.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import os
import secrets

from src.bot.conversation import BotConversation, split_for_telegram
from src.workflows.dialogue_pipeline import DialoguePipeline

logger = logging.getLogger("nexus_studio.runner")

# How long one clip waits for the owner's ✅/🔁/⏹️ before production stops
# on its own (spending nothing more).
REVIEW_TIMEOUT_SECONDS = 6 * 3600


def _markup(buttons):
    if not buttons:
        return None
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in buttons]])


def build_application(bot_token: str, conversation: BotConversation, dialogue_pipeline: DialoguePipeline):
    from telegram import Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

    _background_tasks: set[asyncio.Task] = set()

    def _spawn(coro) -> None:
        task = asyncio.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    # Owner's standing instruction, 2026-10-21 ("اريد الحديث معه كما
    # اتحدث معك ... محادثة حرة كاملة"): every plain text message now
    # routes through a real LLM intent call (BotConversation.handle_message),
    # not silent accumulation. But Telegram's own CLIENT still auto-splits
    # one long typed/pasted message into several separate text updates
    # (the 2026-10-19 finding below) - routing each FRAGMENT through the
    # LLM individually would ask it to interpret a sentence cut off
    # mid-word, and risks misreading a half-sent thought as an
    # instruction. A short per-chat debounce coalesces rapid-fire
    # fragments into one real message before the router ever runs: each
    # new fragment cancels any still-pending timer for that chat and
    # starts a fresh one, so routing only fires once no further fragment
    # has arrived for _TEXT_DEBOUNCE_SECONDS. The task reference is kept
    # in this same dict until it fires or is cancelled - the same
    # strong-reference pattern _spawn uses above, for the same documented
    # asyncio GC risk.
    _TEXT_DEBOUNCE_SECONDS = 1.5
    _debounce_tasks: dict[str, asyncio.Task] = {}

    def _cancel_debounce(chat_id: str) -> None:
        # Real gap found by code review, 2026-10-21, right after the
        # pending_text race fix above: that fix stops the debounce path
        # from ever observing or building from stale/empty state, but it
        # never stopped the debounce task itself from still firing a few
        # moments after the owner's own explicit /new, /full, /edit or
        # /caption already ran. If the owner's command finished first,
        # the debounce call now harmlessly finds nothing to route and
        # sends nothing - but if the command was still in flight (a real,
        # slow reasoning-client call), the debounce call could run
        # concurrently, make its OWN redundant real reasoning-client
        # call on the same pending text, and - however harmlessly to the
        # real screenplay itself - send the owner a second, confusing
        # "أرسل /full أو /new" message right after they just did exactly
        # that. Any explicit command is a clear sign the owner is driving
        # this chat directly right now, so it cancels whatever automatic
        # routing was still waiting - the same reasoning _text() already
        # applies to a NEW fragment superseding an OLDER one, just
        # applied to an explicit command superseding the automatic path
        # entirely.
        existing = _debounce_tasks.pop(chat_id, None)
        if existing is not None:
            existing.cancel()

    async def _notify_or_log(chat_id: str, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
        # Real gap found by code review, 2026-10-21 ("تسحب الرصيد بدون اي
        # فيديو" - the balance gets withdrawn with no video ever arriving):
        # every failure-reporting send_message() in this file used to be
        # called bare. If THAT call itself raises (Telegram rate-limit, a
        # transient network error, anything - the exact ordinary real-world
        # failures this whole file exists to survive), the exception
        # propagates out of a bare asyncio.create_task() with nothing
        # catching it (these tasks are not dispatched through the
        # Application, so add_error_handler never sees them) - total
        # silence, AFTER real fal.ai money may already have been spent.
        # This is the last line of defense: if even this fails, the
        # failure is at least logged (nexus_studio.runner, same pattern as
        # reasoning_client.py's Groq-failure-visibility fix) so the
        # deploy workflow's own log tail can show it, instead of the task
        # vanishing with absolutely no trace anywhere.
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:
            logger.error("Failed to notify chat %s (message not delivered): %s | original text: %s", chat_id, exc, text)

    async def _route_after_debounce(chat_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await asyncio.sleep(_TEXT_DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            return
        _debounce_tasks.pop(chat_id, None)
        # Real gap, caught live, 2026-10-21: this task is created directly
        # with asyncio.create_task (same as _produce_and_notify below), NOT
        # dispatched through python-telegram-bot's own Application - so
        # add_error_handler's global catch (registered below, for the
        # OTHER handlers) never sees an exception raised in here. Without
        # this try/except, any non-DialogueWriterError failure inside
        # handle_message() (a real network error from the reasoning
        # client, for instance - RealGroqReasoningClient/
        # RealGeminiReasoningClient both raise raw exceptions, never
        # DialogueWriterError, on a real API/network failure) would die as
        # an unretrieved Task exception - the exact "owner gets total
        # silence" failure class _produce_and_notify was already fixed for
        # on 2026-10-19, reachable again here through the new router.
        #
        # Second, more severe real gap, caught live minutes later: this
        # called conversation.handle_message() DIRECTLY on the event loop
        # - a real, synchronous reasoning-client HTTP call
        # (RealGroqReasoningClient's own fallback to
        # RealGeminiReasoningClient, whose own retry path calls
        # time.sleep() up to 20s per attempt, 3 attempts). This fires on
        # EVERY debounced free-text message now - far more often than
        # /new or /full ever did - and while it runs, it blocks the
        # ENTIRE bot's event loop, every chat, every handler. The owner
        # hit this directly: pressed the ✅ approve button while a router
        # call was in flight, and Telegram's own real error came back
        # ("Query is too old and response timeout expired or query id is
        # invalid") because the bot could not answer the callback in
        # time - the event loop was fully blocked on this call and never
        # got to it. Same fix _produce_and_notify already uses for its
        # own slow call: run it in the executor so the event loop - and
        # every button press on it - stays free the whole time.
        loop = asyncio.get_running_loop()
        try:
            reply = await loop.run_in_executor(None, conversation.handle_message, chat_id, "")
        except Exception as exc:
            await _notify_or_log(chat_id, context, f"⚠️ تعذّر فهم رسالتك بسبب خطأ تقني: {exc}")
            return
        if reply.text:
            # Real gap found by code review, 2026-10-21: this call used to
            # be unprotected - if _send_reply itself raised (the same
            # Telegram-delivery-failure class _notify_or_log above exists
            # for), the exception died unretrieved in this bare task, and
            # the owner who had just typed something got total silence
            # with no error message at all - the identical shape as the
            # real-money incident, just without money attached here.
            try:
                await _send_reply(chat_id, context, reply)
            except Exception as exc:
                logger.error("Failed to deliver reply to chat %s: %s", chat_id, exc)
                await _notify_or_log(chat_id, context, f"⚠️ تعذّر إرسال الرد إليك بسبب خطأ تقني: {exc}")

    async def _send_reply(chat_id: str, context: ContextTypes.DEFAULT_TYPE, reply) -> None:
        # Real gap found, 2026-10-20 ("لماذا النهاية متقطعة... لماذا لا
        # يكون كله في ميساج واحد؟"): the Bot API's own sendMessage rejects
        # any text over its real 4096-character limit outright - it does
        # not split for you. A long screenplay (many lines/characters) or
        # a produced-result message with several failed/skipped lines
        # could exceed it, and this call used to send reply.text in one
        # single, unprotected call. split_for_telegram() chunks it at real
        # newline boundaries; only the LAST chunk carries the buttons,
        # since those belong on the message the owner actually acts on.
        chunks = split_for_telegram(reply.text)
        for chunk in chunks[:-1]:
            await context.bot.send_message(chat_id=chat_id, text=chunk)
        await context.bot.send_message(chat_id=chat_id, text=chunks[-1], reply_markup=_markup(reply.buttons))
        # Real gap found by code review, 2026-10-21: a failure on clip N
        # used to abort this loop entirely and propagate an exception that
        # _produce_and_notify's except block reported as "فشل إرسال
        # الفيديو إليك" (the video failed to reach you) - false and
        # misleading whenever clips before N had already sent
        # successfully and were genuinely sitting in the owner's chat.
        # Every clip is now attempted regardless of an earlier one's
        # failure, and the failures (if any) are collected and reported
        # by name - never a blanket claim that contradicts what the
        # owner's own chat already shows.
        send_failures: list[str] = []
        if reply.video_path:
            try:
                with open(reply.video_path, "rb") as f:
                    await context.bot.send_video(chat_id=chat_id, video=f)
            except Exception as exc:
                send_failures.append(f"{os.path.basename(reply.video_path)}: {exc}")
        # Real incident, 2026-10-20: when the final merge fails,
        # format_produced() now hands back every real, already-paid clip
        # individually instead of only describing them in text - send each
        # one, or the fix there is inert.
        for clip_path in reply.video_paths:
            try:
                with open(clip_path, "rb") as f:
                    await context.bot.send_video(chat_id=chat_id, video=f)
            except Exception as exc:
                send_failures.append(f"{os.path.basename(clip_path)}: {exc}")
        if send_failures:
            raise RuntimeError("some clips did not reach you: " + "; ".join(send_failures))

    # Owner's choice, 2026-10-26 ("هكذا يعجبني"): every finished clip is
    # sent to the owner with ✅ / 🔁 / ⏹️ and production waits for the tap.
    # token -> (chat_id, future the worker thread is blocked on)
    _pending_reviews: dict[str, tuple[str, concurrent.futures.Future]] = {}

    async def _send_review(chat_id: str, context: ContextTypes.DEFAULT_TYPE, clip_path: str, text: str, buttons) -> None:
        with open(clip_path, "rb") as f:
            await context.bot.send_video(chat_id=chat_id, video=f, caption=text[:1024], reply_markup=_markup(buttons))

    def _make_reviewer(chat_id: str, context: ContextTypes.DEFAULT_TYPE, loop: asyncio.AbstractEventLoop):
        def review(line, clip, attempt: int, can_redo: bool) -> str:
            token = secrets.token_hex(6)
            decision: concurrent.futures.Future = concurrent.futures.Future()
            _pending_reviews[token] = (chat_id, decision)
            buttons = [("✅ قبول", f"clip:accept:{token}")]
            if can_redo:
                buttons.append(("🔁 أعد", f"clip:redo:{token}"))
            buttons.append(("⏹️ أوقف", f"clip:stop:{token}"))
            text = (
                f"🎬 السطر {line.index + 1} - المحاولة {attempt}\n{line.character}: {line.line}\n"
                f"تكلفة هذه اللقطة: ${clip.cost_usd:.3f}"
                + ("\n🔁 الإعادة تُدفع من جديد." if can_redo else "\nهذه آخر محاولة لهذا السطر.")
            )
            try:
                asyncio.run_coroutine_threadsafe(_send_review(chat_id, context, clip.local_path, text, buttons), loop).result(timeout=300)
                try:
                    return decision.result(timeout=REVIEW_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    return "stop"
            finally:
                _pending_reviews.pop(token, None)

        return review

    async def _produce_and_notify(chat_id: str, screenplay, context: ContextTypes.DEFAULT_TYPE) -> None:
        loop = asyncio.get_running_loop()
        try:
            produced = await loop.run_in_executor(
                None,
                functools.partial(dialogue_pipeline.produce, screenplay, clip_reviewer=_make_reviewer(chat_id, context, loop)),
            )
        except Exception as exc:
            await _notify_or_log(
                chat_id, context,
                f"❌ فشل التوليد الحقيقي: {exc}\n\nلم يكتمل أي مقطع - تحقق من رصيد fal.ai الحقيقي قبل إعادة المحاولة.",
            )
            return
        # Real gap found by code review, 2026-10-20 (the owner's own
        # standing demand, "افعل كل شيء لكي لا نخسر المال" - do everything
        # so we don't lose money): production itself was already protected
        # above, but everything AFTER it - building the reply and actually
        # sending the video(s) to Telegram - ran with no error handling at
        # all. If send_video ever raises (Telegram's own ~50MB file-size
        # limit, a network error, anything) after real money was already
        # spent and real clips already exist, the task just dies silently -
        # the exact "owner gets total silence for hours" failure class the
        # 2026-10-19 fix above exists to prevent, reachable again from a
        # different point, and now specifically AFTER real spend instead
        # of before it.
        try:
            reply = conversation.format_produced(chat_id, produced)
            await _send_reply(chat_id, context, reply)
        except Exception as exc:
            await _notify_or_log(
                chat_id, context,
                f"⚠️ اكتمل التوليد الحقيقي (التكلفة ${produced.total_cost_usd:.3f}، "
                f"{len(produced.clips)} مقطع) لكن فشل إرسال الفيديو إليك: {exc}",
            )

    async def _new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Real bug found by code review, 2026-10-21: context.args splits
        # on ANY whitespace, including real newlines - exactly the same
        # defect /edit was fixed for on 2026-10-20, left unfixed here. A
        # story typed directly after "/new 30 ..." (rather than sent as
        # separate plain messages first) had every one of its real
        # newlines silently collapsed into single spaces before reaching
        # handle_new. Same fix: read the raw message text and split only
        # once, so everything after the command - every real newline
        # included - reaches handle_new intact.
        chat_id = str(update.effective_chat.id)
        _cancel_debounce(chat_id)
        raw_text = update.message.text or ""
        parts = raw_text.split(None, 1)
        args_text = parts[1] if len(parts) == 2 else ""
        # Real bug found live, 2026-10-21 ("Query is too old and response
        # timeout expired" on an unrelated button press): handle_new()
        # makes TWO real, synchronous reasoning-client HTTP calls (cast,
        # then lines) - RealGeminiReasoningClient's own retry path can
        # call time.sleep() for up to 20s per attempt, 3 attempts, fully
        # blocking. Called directly like this, that blocks the ENTIRE
        # bot's event loop - every chat, every button press - for the
        # whole duration, not just this one. Same fix _produce_and_notify
        # already uses for its own slow call: run it in the executor so
        # the event loop stays free to answer a callback query (which
        # Telegram invalidates if left unanswered too long) the whole time.
        loop = asyncio.get_running_loop()
        # Real incident, 2026-10-22: conversation.py now wraps every real
        # reasoning-client failure into a clean DialogueWriterError (see
        # dialogue_writer.py's own _complete()), so handle_new() itself
        # no longer raises for a real Groq/Gemini outage - but this
        # try/except is kept anyway, matching _route_after_debounce's own
        # defense-in-depth above, for any OTHER unexpected failure inside
        # handle_new() that is not that specific, already-handled class.
        try:
            reply = await loop.run_in_executor(None, conversation.handle_new, chat_id, args_text)
        except Exception as exc:
            await _notify_or_log(chat_id, context, f"⚠️ تعذّر بناء السيناريو بسبب خطأ تقني: {exc}")
            return
        await _send_reply(chat_id, context, reply)

    async def _full(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Same real bug and same fix as _new above - a story typed
        # directly after "/full" lost every real newline to
        # context.args's whitespace tokenization before this.
        chat_id = str(update.effective_chat.id)
        _cancel_debounce(chat_id)
        raw_text = update.message.text or ""
        parts = raw_text.split(None, 1)
        args_text = parts[1] if len(parts) == 2 else ""
        # Same event-loop-blocking real reasoning-client call, same fix,
        # as _new above.
        loop = asyncio.get_running_loop()
        # Same real incident and same reasoning as _new above.
        try:
            reply = await loop.run_in_executor(None, conversation.handle_full, chat_id, args_text)
        except Exception as exc:
            await _notify_or_log(chat_id, context, f"⚠️ تعذّر بناء السيناريو بسبب خطأ تقني: {exc}")
            return
        await _send_reply(chat_id, context, reply)

    async def _text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Real finding, 2026-10-19: Telegram's own client auto-splits an
        # outgoing message over ~4096 characters into several separate
        # messages - only the first one can carry the /new command, so a
        # long real story needs every later piece collected here instead
        # of being silently dropped. conversation.handle_plain_text()
        # always returns None - no reply - so a multi-message story does
        # not get a chat reply per fragment, only once routing fires.
        #
        # Owner's instruction, 2026-10-21 ("اريد الحديث معه كما اتحدث
        # معك" - free conversation, discussion and collaborative editing):
        # every fragment still accumulates here immediately (so nothing is
        # ever lost even if the process restarts mid-story), but now also
        # (re)starts the debounce timer above. Only once no further
        # fragment arrives for _TEXT_DEBOUNCE_SECONDS does _route_after_debounce
        # actually call the LLM router - see that function's own comment
        # for why a debounce is needed at all.
        chat_id = str(update.effective_chat.id)
        conversation.handle_plain_text(chat_id, update.message.text or "")
        existing = _debounce_tasks.get(chat_id)
        if existing is not None:
            existing.cancel()
        _debounce_tasks[chat_id] = asyncio.create_task(_route_after_debounce(chat_id, context))

    async def _edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Owner's instruction, 2026-10-20: /edit now accepts several
        # "<number> <text>" edits at once, one per real newline - but
        # context.args splits the message on ANY whitespace, including
        # newlines, collapsing multiple edits into one unparseable blob.
        # str.split(None, 1) splits on only the FIRST whitespace run (the
        # one right after "/edit"), so everything after it - including
        # every real newline the owner typed - reaches handle_edit intact.
        chat_id = str(update.effective_chat.id)
        _cancel_debounce(chat_id)
        raw_text = update.message.text or ""
        parts = raw_text.split(None, 1)
        args_text = parts[1] if len(parts) == 2 else ""
        reply = conversation.handle_edit(chat_id, args_text)
        await _send_reply(chat_id, context, reply)

    async def _caption(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        _cancel_debounce(chat_id)
        reply = conversation.handle_caption(chat_id, " ".join(context.args))
        await _send_reply(chat_id, context, reply)

    async def _cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        _cancel_debounce(chat_id)
        reply = conversation.handle_cancel(chat_id)
        await _send_reply(chat_id, context, reply)

    async def _photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Owner's instruction, 2026-10-21 ("الصور ضعها انت الان و هكذا لا
        # ندفع شيء" - you put the photos in yourself, so we don't pay
        # anything): a real photo the owner already has for a character
        # (his own, or any he already paid for elsewhere) can be sent to
        # the bot directly - captioned with that character's EXACT name
        # as it appears in the current screenplay - and is saved straight
        # into DialoguePipeline's own real cache path. The next real
        # production of this screenplay then finds it already cached,
        # the exact same way a previously-generated photo would be: zero
        # cost, zero real image-generation call, no fal.ai/Replicate
        # request made for this character at all.
        chat_id = str(update.effective_chat.id)
        character_name = (update.message.caption or "").strip()
        if not character_name:
            await _notify_or_log(
                chat_id, context,
                "أرسل اسم الشخصية ككابشن (caption) مع الصورة - بالضبط كما ظهر في السيناريو الحالي.",
            )
            return
        screenplay = conversation.get_current_screenplay(chat_id)
        if screenplay is None:
            await _notify_or_log(chat_id, context, "لا يوجد سيناريو حالي لحفظ صورة له - أرسل /new أو /full أولاً.")
            return
        if not any(c.name == character_name for c in screenplay.characters):
            known_names = "، ".join(c.name for c in screenplay.characters)
            await _notify_or_log(
                chat_id, context,
                f"لم أجد شخصية باسم '{character_name}' في السيناريو الحالي. الأسماء الموجودة: {known_names}",
            )
            return
        if dialogue_pipeline is None:
            await _notify_or_log(chat_id, context, "لا يوجد اتصال حقيقي بخط الإنتاج لحفظ الصورة.")
            return
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = bytes(await photo_file.download_as_bytearray())
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, dialogue_pipeline.save_character_photo, screenplay.title, character_name, photo_bytes
            )
        except Exception as exc:
            await _notify_or_log(chat_id, context, f"⚠️ فشل حفظ الصورة: {exc}")
            return
        await _notify_or_log(
            chat_id, context,
            f"✅ تم حفظ صورة '{character_name}' - لن تُولَّد ولن يُدفع أي مبلغ لصورتها بعد الآن في هذا السيناريو.",
        )

    async def _button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        chat_id = str(update.effective_chat.id)
        _cancel_debounce(chat_id)
        data = query.data

        if data.startswith("clip:"):
            _, decision, token = data.split(":", 2)
            pending = _pending_reviews.get(token)
            if pending is None or pending[0] != chat_id or pending[1].done():
                await _notify_or_log(chat_id, context, "هذه اللقطة لم تعد تنتظر قرارك.")
                return
            pending[1].set_result(decision)
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception as exc:
                logger.warning("could not remove review buttons in chat %s: %s", chat_id, exc)
            labels = {"accept": "✅ قُبلت. ننتقل للسطر التالي.", "redo": "🔁 تُعاد هذه اللقطة الآن.", "stop": "⏹️ توقف الإنتاج. لن يُصرف شيء آخر."}
            await _notify_or_log(chat_id, context, labels.get(decision, decision))
        elif data == "approve":
            reply, screenplay = conversation.handle_approve(chat_id)
            await _send_reply(chat_id, context, reply)
            if screenplay is not None:
                _spawn(_produce_and_notify(chat_id, screenplay, context))
        elif data in ("cancel", "discard"):
            reply = conversation.handle_cancel(chat_id)
            await _send_reply(chat_id, context, reply)
        elif data.startswith("publish:"):
            # Real bug found by code review, 2026-10-21, same pattern as
            # /new, /full and the debounce router above: handle_publish()
            # calls PublishingAgent.publish_episode(), which makes a real,
            # synchronous upload_video() HTTP call - a real video file
            # upload to YouTube, which can genuinely take a long time for
            # anything but a tiny clip. Called directly here, that blocks
            # the whole bot's event loop - every other chat, every button
            # press, including another chat's own Approve - for the whole
            # upload. Same fix, same reason: run it in the executor.
            platform = data.split(":", 1)[1]
            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(None, conversation.handle_publish, chat_id, platform)
            await _send_reply(chat_id, context, reply)

    async def _on_error(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = None
        if isinstance(update, Update) and update.effective_chat is not None:
            chat_id = update.effective_chat.id
        if chat_id is None:
            return
        # Real gap found by code review, 2026-10-21: this used to swallow
        # a failure of the notification itself with a bare `except: pass`
        # - not even logged. This is the Application's own LAST-RESORT
        # handler, reachable from every CommandHandler/CallbackQueryHandler
        # (/new, /full, /edit, /caption, the approve/publish buttons) -
        # silently dropping its own failure here is the same "total
        # silence" shape _notify_or_log above exists to prevent, just in
        # the one place meant to be the final safety net.
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ حدث خطأ حقيقي غير متوقع: {context.error}")
        except Exception as exc:
            logger.error("Failed to notify chat %s of error %s: %s", chat_id, context.error, exc)

    application = Application.builder().token(bot_token).build()
    application.add_error_handler(_on_error)
    application.add_handler(CommandHandler("new", _new))
    application.add_handler(CommandHandler("full", _full))
    application.add_handler(CommandHandler("edit", _edit))
    application.add_handler(CommandHandler("caption", _caption))
    application.add_handler(CommandHandler("cancel", _cancel))
    application.add_handler(CallbackQueryHandler(_button))
    application.add_handler(MessageHandler(filters.PHOTO, _photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _text))
    return application
