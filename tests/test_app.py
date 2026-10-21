from __future__ import annotations

from src.app import build_app
from src.config import Settings
from src.integrations.fal_video_client import REAL_BILLED_PRICE_PER_SECOND_USD, RealFalVideoClient
from src.integrations.replicate_video_client import PROVISIONAL_PRICE_PER_SECOND_USD, RealReplicateVideoClient


def _settings(**overrides) -> Settings:
    base = dict(
        groq_api_key="", groq_model="m", gemini_api_key="", gemini_model="m",
        youtube_client_id="", youtube_client_secret="", youtube_refresh_token="",
        telegram_bot_token="tg", azure_speech_key="azure-key", azure_speech_region="eastus",
        azure_tts_lang="ar-SY", fal_key="", replicate_api_token="", video_provider="",
        output_dir="./output-test", max_episode_cost_usd=10.0,
    )
    base.update(overrides)
    return Settings(**base)


def test_build_app_auto_selects_replicate_when_only_replicate_is_configured():
    app = build_app(_settings(replicate_api_token="r8_real"))

    assert isinstance(app.fal_video, RealReplicateVideoClient)
    assert app.dialogue_pipeline._price_per_second_usd == PROVISIONAL_PRICE_PER_SECOND_USD
    assert app.conversation._price_per_second_usd == PROVISIONAL_PRICE_PER_SECOND_USD


def test_build_app_auto_selects_fal_when_only_fal_is_configured():
    app = build_app(_settings(fal_key="fal-real"))

    assert isinstance(app.fal_video, RealFalVideoClient)
    assert app.dialogue_pipeline._price_per_second_usd == REAL_BILLED_PRICE_PER_SECOND_USD
    assert app.conversation._price_per_second_usd == REAL_BILLED_PRICE_PER_SECOND_USD


def test_build_app_prefers_replicate_when_both_are_configured_and_no_explicit_choice():
    # Owner's instruction, 2026-10-21 ("الان اريد البوت بربليكات"): with
    # no explicit VIDEO_PROVIDER, Replicate must win when both real keys
    # are present - the provider he asked to move to, not whichever
    # happens to be checked first in the code.
    app = build_app(_settings(fal_key="fal-real", replicate_api_token="r8_real"))

    assert isinstance(app.fal_video, RealReplicateVideoClient)


def test_build_app_honors_an_explicit_video_provider_override():
    # Even with Replicate configured too, an explicit VIDEO_PROVIDER=fal
    # must still pick fal.ai - the owner's own explicit choice always
    # wins over the auto-select default, the same principle this project
    # already applies to explicit commands overriding automatic routing.
    app = build_app(_settings(fal_key="fal-real", replicate_api_token="r8_real", video_provider="fal"))

    assert isinstance(app.fal_video, RealFalVideoClient)


def test_build_app_has_no_real_pipeline_when_neither_video_provider_is_configured():
    app = build_app(_settings())

    assert app.fal_video is None
    assert app.dialogue_pipeline is None
    assert app.conversation is None
