"""Long-running service entrypoint - `python -m src.main` (also the
Docker CMD). Blocks on the Telegram bot's long-polling loop, same shape
as nexus-commerce's own src/main.py.
"""
from __future__ import annotations

import logging

from .app import build_app
from .bot.runner import build_application
from .config import Settings

logging.basicConfig(level=logging.INFO)
# Real secret leak found in a GitHub Actions deploy log, 2026-10-20: httpx
# logs the full request URL at INFO level, and the Telegram Bot API URL
# embeds the live bot token (https://api.telegram.org/bot<TOKEN>/getMe) -
# that line reached a real CI run's own (non-secret-masked) log untouched.
# The token is the one credential that lets anyone impersonate this bot, so
# httpx's own logger is held to WARNING regardless of the app's level.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("nexus_studio.main")


def main() -> None:
    settings = Settings.from_env()
    app = build_app(settings)

    if not settings.telegram_configured:
        logger.error("TELEGRAM_BOT_TOKEN not set - the bot has nothing to listen on, exiting")
        return

    if app.conversation is None or app.dialogue_pipeline is None:
        logger.error("FAL_KEY and/or AZURE_SPEECH_KEY not set - the real dialogue pipeline is unavailable, exiting")
        return

    application = build_application(settings.telegram_bot_token, app.conversation, app.dialogue_pipeline)
    logger.info("nexus-studio bot starting (long-polling)")
    application.run_polling()


if __name__ == "__main__":
    main()
