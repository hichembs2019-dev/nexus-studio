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

EpisodePipeline.produce() is slow (minutes) and synchronous; running it
directly inside an async handler would block the whole bot. It runs in
the default executor (a background thread) instead, so /new, /edit and
button presses on OTHER chats keep working while a generation is in
flight.
"""
from __future__ import annotations

import asyncio

from src.bot.conversation import BotConversation
from src.workflows.episode_pipeline import EpisodePipeline


def _markup(buttons):
    if not buttons:
        return None
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in buttons]])


def build_application(bot_token: str, conversation: BotConversation, pipeline: EpisodePipeline):
    from telegram import Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

    async def _send_reply(chat_id: str, context: ContextTypes.DEFAULT_TYPE, reply) -> None:
        await context.bot.send_message(chat_id=chat_id, text=reply.text, reply_markup=_markup(reply.buttons))
        if reply.video_path:
            with open(reply.video_path, "rb") as f:
                await context.bot.send_video(chat_id=chat_id, video=f)

    async def _produce_and_notify(chat_id: str, episode, context: ContextTypes.DEFAULT_TYPE) -> None:
        loop = asyncio.get_running_loop()
        produced = await loop.run_in_executor(None, pipeline.produce, episode)
        reply = conversation.format_produced(chat_id, produced)
        await _send_reply(chat_id, context, reply)

    async def _new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = str(update.effective_chat.id)
        reply = conversation.handle_new(chat_id, " ".join(context.args))
        await _send_reply(chat_id, context, reply)

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
            reply, episode = conversation.handle_approve(chat_id)
            await _send_reply(chat_id, context, reply)
            if episode is not None:
                asyncio.create_task(_produce_and_notify(chat_id, episode, context))
        elif data in ("cancel", "discard"):
            reply = conversation.handle_cancel(chat_id)
            await _send_reply(chat_id, context, reply)
        elif data.startswith("publish:"):
            platform = data.split(":", 1)[1]
            reply = conversation.handle_publish(chat_id, platform)
            await _send_reply(chat_id, context, reply)

    application = Application.builder().token(bot_token).build()
    application.add_handler(CommandHandler("new", _new))
    application.add_handler(CommandHandler("edit", _edit))
    application.add_handler(CommandHandler("cancel", _cancel))
    application.add_handler(CallbackQueryHandler(_button))
    return application
