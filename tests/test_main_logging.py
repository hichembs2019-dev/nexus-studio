from __future__ import annotations

import importlib
import logging


def test_httpx_logger_is_not_info_level():
    """Real finding, 2026-10-20: a GitHub Actions deploy log showed the
    live Telegram bot token in plain text, because httpx logs the full
    request URL (which embeds the token) at INFO, and src.main's own
    logging.basicConfig(level=logging.INFO) left it there. Importing
    src.main must push httpx's own logger below INFO regardless of
    whatever level the root logger is configured at.
    """
    import src.main  # noqa: F401

    importlib.reload(src.main)

    httpx_logger = logging.getLogger("httpx")
    assert httpx_logger.level >= logging.WARNING, (
        f"httpx logger is at {logging.getLevelName(httpx_logger.level)} - "
        "its INFO logs include the live bot token in the request URL"
    )
