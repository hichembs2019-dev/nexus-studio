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

from src.bot.conversation import BotConversation, split_for_telegram
from src.workflows.dialogue_pipeline import DialoguePipeline


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

    async def _route_after_debounce(chat_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await asyncio.sleep(_TEXT_DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            return
        _debounce_tasks.pop(chat_id, None)
        reply = conversation.handle_message(chat_id, "")
        if reply.text:
            await _send_reply(chat_id, context, reply)

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
        if reply.video_path:
            with open(reply.video_path, "rb") as f:
                await context.bot.send_video(chat_id=chat_id, video=f)
        # Real incident, 2026-10-20: when the final merge fails,
        # format_produced() now hands back every real, already-paid clip
        # individually instead of only describing them in text - send each
        # one, or the fix there is inert.
        for clip_path in reply.video_paths:
            with open(clip_path, "rb") as f:
                await context.bot.send_video(chat_id=chat_id, video=f)

    async def _produce_and_notify(chat_id: str, screenplay, context: ContextTypes.DEFAULT_TYPE) -> None:
        loop = asyncio.get_running_loop()
        try:
            produced = await loop.run_in_executor(None, dialogue_pipeline.produce, screenplay)
        except Exception as exc:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ فشل التوليد الحقيقي: {exc}\n\nلم يكتمل أي مقطع - تحقق من رصيد fal.ai الحقيقي قبل إعادة المحاولة.",
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
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"⚠️ اكتمل التوليد الحقيقي (التكلفة ${produced.total_cost_usd:.3f}، "
                    f"{len(produced.clips)} مقطع) لكن فشل إرسال الفيديو إليك: {exc}"
                ),
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
        raw_text = update.message.text or ""
        parts = raw_text.split(None, 1)
        args_text = parts[1] if len(parts) == 2 else ""
        reply = conversation.handle_new(chat_id, args_text)
        await _send_reply(chat_id, context, reply)

    async def _full(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Same real bug and same fix as _new above - a story typed
        # directly after "/full" lost every real newline to
        # context.args's whitespace tokenization before this.
        chat_id = str(update.effective_chat.id)
        raw_text = update.message.text or ""
        parts = raw_text.split(None, 1)
        args_text = parts[1] if len(parts) == 2 else ""
        reply = conversation.handle_full(chat_id, args_text)
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
        raw_text = update.message.text or ""
        parts = raw_text.split(None, 1)
        args_text = parts[1] if len(parts) == 2 else ""
        reply = conversation.handle_edit(chat_id, args_text)
        await _send_reply(chat_id, context, reply)

    async def _caption(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        reply = conversation.handle_caption(chat_id, " ".join(context.args))
        await _send_reply(chat_id, context, reply)

    async def _cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        reply = conversation.handle_cancel(chat_id)
        await _send_reply(chat_id, context, reply)

    async def _button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        chat_id = str(update.effective_chat.id)
        data = query.data

        if data == "approve":
            reply, screenplay = conversation.handle_approve(chat_id)
            await _send_reply(chat_id, context, reply)
            if screenplay is not None:
                _spawn(_produce_and_notify(chat_id, screenplay, context))
        elif data in ("cancel", "discard"):
            reply = conversation.handle_cancel(chat_id)
            await _send_reply(chat_id, context, reply)
        elif data.startswith("publish:"):
            platform = data.split(":", 1)[1]
            reply = conversation.handle_publish(chat_id, platform)
            await _send_reply(chat_id, context, reply)

    async def _on_error(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = None
        if isinstance(update, Update) and update.effective_chat is not None:
            chat_id = update.effective_chat.id
        if chat_id is None:
            return
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ حدث خطأ حقيقي غير متوقع: {context.error}")
        except Exception:
            pass

    application = Application.builder().token(bot_token).build()
    application.add_error_handler(_on_error)
    application.add_handler(CommandHandler("new", _new))
    application.add_handler(CommandHandler("full", _full))
    application.add_handler(CommandHandler("edit", _edit))
    application.add_handler(CommandHandler("caption", _caption))
    application.add_handler(CommandHandler("cancel", _cancel))
    application.add_handler(CallbackQueryHandler(_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _text))
    return application
