"""Thin, real-integration-only wiring between python-telegram-bot and
BotConversation - the same split nexus-commerce's own
src/integrations/telegram_listener.py documents and uses: the decision
logic (conversation.py) is pure and unit-tested, this file needs a real
bot token and a live Telegram connection to exercise at all, so it is
kept as small as possible.

CommandHandler's constructor and context.args shape are verified from
python-telegram-bot's own docs (2026-10-17). send_video/reply_video are
not independently doc-verified beyond what nexus-studio's own
telegram_client.py already notes (send_message/InlineKeyboardMarkup are
reused from nexus-commerce's verified usage) - confirm on first live run.

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
"""
from __future__ import annotations

import asyncio

from src.bot.conversation import BotConversation
from src.workflows.dialogue_pipeline import DialoguePipeline


def _markup(buttons):
    if not buttons:
        return None
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in buttons]])


def build_application(bot_token: str, conversation: BotConversation, dialogue_pipeline: DialoguePipeline):
    from telegram import Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

    async def _send_reply(chat_id: str, context: ContextTypes.DEFAULT_TYPE, reply) -> None:
        await context.bot.send_message(chat_id=chat_id, text=reply.text, reply_markup=_markup(reply.buttons))
        if reply.video_path:
            with open(reply.video_path, "rb") as f:
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
        reply = conversation.format_produced(chat_id, produced)
        await _send_reply(chat_id, context, reply)

    async def _new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        reply = conversation.handle_new(chat_id, " ".join(context.args))
        await _send_reply(chat_id, context, reply)

    async def _full(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        reply = conversation.handle_full(chat_id, " ".join(context.args))
        await _send_reply(chat_id, context, reply)

    async def _text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Real finding, 2026-10-19: Telegram's own client auto-splits an
        # outgoing message over ~4096 characters into several separate
        # messages - only the first one can carry the /new command, so a
        # long real story needs every later piece collected here instead
        # of being silently dropped. conversation.handle_plain_text()
        # always returns None - no reply - so a multi-message story does
        # not get a chat reply per fragment, only once /new is sent.
        chat_id = str(update.effective_chat.id)
        conversation.handle_plain_text(chat_id, update.message.text or "")

    async def _edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        reply = conversation.handle_edit(chat_id, " ".join(context.args))
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
                asyncio.create_task(_produce_and_notify(chat_id, screenplay, context))
        elif data in ("cancel", "discard"):
            reply = conversation.handle_cancel(chat_id)
            await _send_reply(chat_id, context, reply)
        elif data.startswith("publish:"):
            platform = data.split(":", 1)[1]
            reply = conversation.handle_publish(chat_id, platform)
            await _send_reply(chat_id, context, reply)

    application = Application.builder().token(bot_token).build()
    application.add_handler(CommandHandler("new", _new))
    application.add_handler(CommandHandler("full", _full))
    application.add_handler(CommandHandler("edit", _edit))
    application.add_handler(CommandHandler("cancel", _cancel))
    application.add_handler(CallbackQueryHandler(_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _text))
    return application
