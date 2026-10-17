"""Single source of real-vs-mock decisions, mirroring nexus-commerce's own
convention: a missing secret means "use the mock for that piece and keep
going", never a crash.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    comfyui_base_url: str
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    jupyter_base_url: str
    jupyter_password: str
    youtube_client_id: str
    youtube_client_secret: str
    youtube_refresh_token: str
    telegram_bot_token: str
    output_dir: str

    @property
    def comfyui_configured(self) -> bool:
        return bool(self.comfyui_base_url)

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def jupyter_configured(self) -> bool:
        return bool(self.jupyter_base_url and self.jupyter_password)

    @property
    def youtube_configured(self) -> bool:
        return bool(self.youtube_client_id and self.youtube_client_secret and self.youtube_refresh_token)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            comfyui_base_url=os.environ.get("COMFYUI_BASE_URL", ""),
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it"),
            jupyter_base_url=os.environ.get("JUPYTER_BASE_URL", ""),
            jupyter_password=os.environ.get("JUPYTER_PASSWORD", ""),
            youtube_client_id=os.environ.get("YOUTUBE_CLIENT_ID", ""),
            youtube_client_secret=os.environ.get("YOUTUBE_CLIENT_SECRET", ""),
            youtube_refresh_token=os.environ.get("YOUTUBE_REFRESH_TOKEN", ""),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            output_dir=os.environ.get("OUTPUT_DIR", "./output"),
        )
