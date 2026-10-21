"""Single source of real-vs-mock decisions, mirroring nexus-commerce's own
convention: a missing secret means "use the mock for that piece and keep
going", never a crash.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    youtube_client_id: str
    youtube_client_secret: str
    youtube_refresh_token: str
    telegram_bot_token: str
    azure_speech_key: str
    azure_speech_region: str
    azure_tts_lang: str
    fal_key: str
    # Owner's instruction, 2026-10-21 ("الان اريد البوت بربليكات" - connect
    # the bot to Replicate now), after a real, owner-run test on
    # Replicate confirmed its billing page matches its quoted price
    # exactly (unlike fal.ai's own real invoice, see docs/SOURCES.md):
    # a second real video provider, wired in alongside fal.ai rather
    # than replacing it outright, so either can be selected without a
    # code change.
    replicate_api_token: str
    # Explicit override: "fal" or "replicate". Empty string means
    # auto-select (see build_app() in app.py) - prefers Replicate when
    # it is configured, since that is the provider the owner asked to
    # move to; falls back to fal.ai otherwise.
    video_provider: str
    output_dir: str
    # Owner's instruction, 2026-10-20 ("قلت لك ضع له كود يشتغل و يسحب
    # المال الا عند صنع فيديو" - I told you to put code in that only
    # draws money when a video is actually made): fal.ai itself already
    # only bills for real completed output (confirmed live: a job that
    # times out costs $0.000, see docs/SOURCES.md "fal.ai job timeout").
    # This is a SEPARATE, additional guard: a hard ceiling on total real
    # spend within one production run, so a long or malformed screenplay
    # can never silently run the real bill past what the owner is willing
    # to lose in one sitting, independent of the cost estimate already
    # shown before Approve.
    max_episode_cost_usd: float

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def youtube_configured(self) -> bool:
        return bool(self.youtube_client_id and self.youtube_client_secret and self.youtube_refresh_token)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token)

    @property
    def azure_speech_configured(self) -> bool:
        return bool(self.azure_speech_key and self.azure_speech_region)

    @property
    def fal_configured(self) -> bool:
        return bool(self.fal_key)

    @property
    def replicate_configured(self) -> bool:
        return bool(self.replicate_api_token)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it"),
            youtube_client_id=os.environ.get("YOUTUBE_CLIENT_ID", ""),
            youtube_client_secret=os.environ.get("YOUTUBE_CLIENT_SECRET", ""),
            youtube_refresh_token=os.environ.get("YOUTUBE_REFRESH_TOKEN", ""),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            azure_speech_key=os.environ.get("AZURE_SPEECH_KEY", ""),
            azure_speech_region=os.environ.get("AZURE_SPEECH_REGION", ""),
            # Owner's instruction, 2026-10-19: Syrian ("neutral/white")
            # Arabic, widely understood across North Africa, the same
            # dialect used dubbing Turkish series - what the real
            # DialoguePipeline path actually uses (see character_photos.py).
            azure_tts_lang=os.environ.get("AZURE_TTS_LANG", "ar-SY"),
            fal_key=os.environ.get("FAL_KEY", ""),
            replicate_api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
            video_provider=os.environ.get("VIDEO_PROVIDER", ""),
            output_dir=os.environ.get("OUTPUT_DIR", "./output"),
            max_episode_cost_usd=float(os.environ.get("MAX_EPISODE_COST_USD", "10.0")),
        )
