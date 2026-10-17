"""Sending/editing Telegram messages. The python-telegram-bot shapes
below (`Bot(token=)`, `bot.send_message(chat_id=, text=, reply_markup=)`,
`InlineKeyboardMarkup`/`InlineKeyboardButton`) are the exact ones already
verified for nexus-commerce (see that project's
src/integrations/telegram_client.py and docs/SOURCES.md "Telegram") -
reused here as already-verified facts. `send_video` and
`answer_callback_query` are new to this project; their Bot API wire
signatures were not independently re-read from Telegram's own docs line
by line (WebFetch truncated both the python-telegram-bot and
core.telegram.org pages before reaching the parameter tables) - the
50MB-per-video cap for bot uploads is corroborated by multiple
third-party docs mirrors, not Telegram's own page directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TelegramClient(ABC):
    @abstractmethod
    def send_message(self, chat_id: str, text: str, buttons: list[tuple[str, str]] | None = None) -> str:
        """Sends `text`, optionally with inline buttons (label, callback_data
        pairs) in a single row. Returns the sent message's id."""

    @abstractmethod
    def send_video(self, chat_id: str, video_path: str, caption: str) -> str:
        """Uploads and sends the real video file at video_path. Returns the
        sent message's id."""

    @abstractmethod
    def answer_callback_query(self, callback_query_id: str) -> None:
        """Acknowledges a button press so Telegram stops showing the
        client-side loading spinner on it."""


class RealTelegramClient(TelegramClient):
    def __init__(self, bot_token: str):
        self._bot_token = bot_token

    def send_message(self, chat_id: str, text: str, buttons: list[tuple[str, str]] | None = None) -> str:
        import asyncio

        from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

        reply_markup = None
        if buttons:
            reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in buttons]])

        async def _send() -> str:
            bot = Bot(token=self._bot_token)
            message = await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
            return str(message.message_id)

        return asyncio.run(_send())

    def send_video(self, chat_id: str, video_path: str, caption: str) -> str:
        import asyncio

        from telegram import Bot

        async def _send() -> str:
            bot = Bot(token=self._bot_token)
            with open(video_path, "rb") as f:
                message = await bot.send_video(chat_id=chat_id, video=f, caption=caption)
            return str(message.message_id)

        return asyncio.run(_send())

    def answer_callback_query(self, callback_query_id: str) -> None:
        import asyncio

        from telegram import Bot

        async def _answer() -> None:
            bot = Bot(token=self._bot_token)
            await bot.answer_callback_query(callback_query_id)

        asyncio.run(_answer())


class MockTelegramClient(TelegramClient):
    def __init__(self):
        self.sent_messages: list[dict] = []
        self.sent_videos: list[dict] = []
        self.answered_callbacks: list[str] = []
        self._next_message_id = 1

    def send_message(self, chat_id: str, text: str, buttons: list[tuple[str, str]] | None = None) -> str:
        self.sent_messages.append({"chat_id": chat_id, "text": text, "buttons": buttons})
        message_id = str(self._next_message_id)
        self._next_message_id += 1
        return message_id

    def send_video(self, chat_id: str, video_path: str, caption: str) -> str:
        self.sent_videos.append({"chat_id": chat_id, "video_path": video_path, "caption": caption})
        message_id = str(self._next_message_id)
        self._next_message_id += 1
        return message_id

    def answer_callback_query(self, callback_query_id: str) -> None:
        self.answered_callbacks.append(callback_query_id)
