from __future__ import annotations

from src.app import build_app
from src.config import Settings
from src.integrations.fal_video_client import REAL_BILLED_PRICE_PER_SECOND_USD, RealFalVideoClient
from src.integrations.image_client import RealFalImageClient
from src.integrations.replicate_image_client import RealReplicateImageClient
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


def test_build_app_never_constructs_a_fal_client_when_replicate_is_chosen():
    # Owner's standing instruction, 2026-10-21 ("لا تضع السراق ... احذفهم
    # من الكود" - don't put the thieves [fal.ai] in, remove them from
    # the code): with Replicate chosen (even when FAL_KEY is ALSO set,
    # as it genuinely is on the real deployment - fal.ai must never be
    # constructed or called for EITHER video or the character portrait.
    app = build_app(_settings(fal_key="fal-real", replicate_api_token="r8_real"))

    assert isinstance(app.fal_video, RealReplicateVideoClient)
    assert isinstance(app.dialogue_pipeline._image_client, RealReplicateImageClient)
    assert not isinstance(app.fal_video, RealFalVideoClient)
    assert not isinstance(app.dialogue_pipeline._image_client, RealFalImageClient)


def test_build_app_uses_fal_for_images_too_when_fal_is_explicitly_chosen():
    app = build_app(_settings(fal_key="fal-real", replicate_api_token="r8_real", video_provider="fal"))

    assert isinstance(app.dialogue_pipeline._image_client, RealFalImageClient)


def test_build_app_has_no_real_pipeline_when_neither_video_provider_is_configured():
    app = build_app(_settings())

    assert app.fal_video is None
    assert app.dialogue_pipeline is None
    assert app.conversation is None
