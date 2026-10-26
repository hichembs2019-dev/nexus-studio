from __future__ import annotations
import pytest

from src.app import build_app
from src.config import Settings
from src.integrations.replicate_image_client import PRICE_PER_IMAGE_USD_UNVERIFIED, RealReplicateImageClient
from src.integrations.replicate_video_client import PROVISIONAL_PRICE_PER_SECOND_USD, RealReplicateVideoClient
from src.integrations.self_hosted_video_client import SelfHostedS2VVideoClient, self_hosted_price_per_second_usd


def _settings(**overrides) -> Settings:
    base = dict(
        groq_api_key="", groq_model="m", gemini_api_key="", gemini_model="m",
        youtube_client_id="", youtube_client_secret="", youtube_refresh_token="",
        telegram_bot_token="tg", azure_speech_key="azure-key", azure_speech_region="eastus",
        azure_tts_lang="ar-SY", replicate_api_token="",
        runpod_api_key="", runpod_s2v_pod_id="", runpod_s2v_hourly_rate_usd=0.59,
        multitalk_enabled=False,
        output_dir="./output-test", max_episode_cost_usd=10.0,
    )
    base.update(overrides)
    return Settings(**base)


def test_build_app_wires_the_real_replicate_clients_when_configured():
    app = build_app(_settings(replicate_api_token="r8_real"))

    assert isinstance(app.video_client, RealReplicateVideoClient)
    assert isinstance(app.dialogue_pipeline._image_client, RealReplicateImageClient)
    assert app.dialogue_pipeline._price_per_second_usd == PROVISIONAL_PRICE_PER_SECOND_USD
    assert app.conversation._price_per_second_usd == PROVISIONAL_PRICE_PER_SECOND_USD
    # Owner's cleanup instruction, 2026-10-21: the portrait price must
    # follow the real wired-in provider exactly like the video price does -
    # found to be silently hard-coded to fal.ai's own constant instead,
    # before fal.ai was deleted entirely.
    assert app.dialogue_pipeline._price_per_image_usd == PRICE_PER_IMAGE_USD_UNVERIFIED
    assert app.conversation._price_per_image_usd == PRICE_PER_IMAGE_USD_UNVERIFIED


def test_build_app_wires_replicates_own_image_price_not_a_stale_constant(monkeypatch):
    # Real gap caught in this exact review: a test that only compares
    # against PRICE_PER_IMAGE_USD_UNVERIFIED's current value would still
    # pass even if app.py wired in the wrong constant by mistake - proven
    # here by monkeypatching Replicate's constant to a value nothing else
    # in this codebase could coincidentally equal, so only correct wiring
    # can pass this. Patched where app.py actually looks it up (its own
    # `from ... import` binding), not in the original defining module -
    # that import already happened when this test file loaded src.app.
    import src.app as app_module

    monkeypatch.setattr(app_module, "PRICE_PER_IMAGE_USD_UNVERIFIED", 0.4242)

    app = build_app(_settings(replicate_api_token="r8_real"))

    assert app.dialogue_pipeline._price_per_image_usd == 0.4242
    assert app.conversation._price_per_image_usd == 0.4242


def test_build_app_has_no_real_pipeline_when_replicate_is_not_configured():
    app = build_app(_settings())

    assert app.video_client is None
    assert app.dialogue_pipeline is None
    assert app.conversation is None


def test_build_app_prefers_the_self_hosted_s2v_engine_when_a_real_pod_is_configured():
    app = build_app(_settings(
        replicate_api_token="r8_real", runpod_api_key="rp_real", runpod_s2v_pod_id="wt5ennaph7w5nh",
    ))

    assert isinstance(app.video_client, SelfHostedS2VVideoClient)
    # Portraits still go through Replicate - there is no self-hosted image
    # pipeline built (see app.py's own comment on this branch).
    assert isinstance(app.dialogue_pipeline._image_client, RealReplicateImageClient)
    expected_rate = self_hosted_price_per_second_usd(0.59)
    assert app.dialogue_pipeline._price_per_second_usd == expected_rate
    assert app.conversation._price_per_second_usd == expected_rate


def test_build_app_falls_back_to_replicate_video_when_no_real_pod_is_configured():
    # Mutation check: a version that always preferred self-hosted
    # regardless of settings.self_hosted_s2v_configured would still pass
    # test_build_app_wires_the_real_replicate_clients_when_configured
    # above by accident if that test's own assertion were loosened -
    # this one pins the OTHER side of the same branch explicitly.
    app = build_app(_settings(replicate_api_token="r8_real", runpod_api_key="", runpod_s2v_pod_id=""))

    assert isinstance(app.video_client, RealReplicateVideoClient)
    assert app.dialogue_pipeline._price_per_second_usd == PROVISIONAL_PRICE_PER_SECOND_USD


def test_build_app_does_not_use_self_hosted_s2v_without_a_real_replicate_token_for_portraits():
    # Real constraint: there is no self-hosted image pipeline, so a real
    # Runpod pod alone (no REPLICATE_API_TOKEN) must not pick the
    # self-hosted branch - it would have a video provider but no way to
    # ever produce a character portrait.
    app = build_app(_settings(replicate_api_token="", runpod_api_key="rp_real", runpod_s2v_pod_id="wt5ennaph7w5nh"))

    assert app.video_client is None


def test_build_app_leaves_multitalk_off_by_default_even_with_a_real_pod_configured():
    # Owner's own standing rule (Settings.multitalk_enabled's own
    # docstring): a real, newly-built, real-money-capable path must never
    # go live just because its prerequisites happen to be configured -
    # only the owner's own explicit MULTITALK_ENABLED switch turns it on.
    app = build_app(_settings(
        replicate_api_token="r8_real", runpod_api_key="rp_real", runpod_s2v_pod_id="wt5ennaph7w5nh",
        multitalk_enabled=False,
    ))

    assert app.dialogue_pipeline._multitalk_client is None


def test_build_app_wires_the_real_multitalk_client_once_explicitly_enabled():
    from src.integrations.multitalk_video_client import MultiTalkVideoClient

    app = build_app(_settings(
        replicate_api_token="r8_real", runpod_api_key="rp_real", runpod_s2v_pod_id="wt5ennaph7w5nh",
        multitalk_enabled=True,
    ))

    assert isinstance(app.dialogue_pipeline._multitalk_client, MultiTalkVideoClient)


def test_build_app_never_enables_multitalk_without_a_real_self_hosted_pod():
    # multitalk_enabled alone (e.g. a stray env var with no real Runpod
    # pod configured) must not wire a client with no real pod id to call.
    app = build_app(_settings(replicate_api_token="r8_real", runpod_api_key="", runpod_s2v_pod_id="", multitalk_enabled=True))

    assert app.dialogue_pipeline is None or app.dialogue_pipeline._multitalk_client is None


def test_gemini_writes_the_screenplay_with_groq_as_its_fallback():
    from src.integrations.reasoning_client import RealGeminiReasoningClient, RealGroqReasoningClient

    app = build_app(_settings(replicate_api_token="r8_real", groq_api_key="gsk", gemini_api_key="gem"))

    writer_client = app.conversation._dialogue_writer._reasoning_client
    assert isinstance(writer_client, RealGeminiReasoningClient)
    assert isinstance(writer_client._fallback, RealGroqReasoningClient)


def test_build_app_prefers_kling_only_when_the_owner_turned_it_on():
    from src.integrations.kling_video_client import KlingVideoClient, estimated_price_per_second_usd

    pod = dict(replicate_api_token="r8_real", runpod_api_key="rp", runpod_s2v_pod_id="pod1")
    # A key alone changes nothing.
    app = build_app(_settings(**pod, kling_api_key="k"))
    assert isinstance(app.video_client, SelfHostedS2VVideoClient)

    app = build_app(_settings(**pod, kling_api_key="k", kling_video_enabled=True, kling_unit_price_usd=0.098))
    assert isinstance(app.video_client, KlingVideoClient)
    assert app.dialogue_pipeline._multitalk_client is None
    assert app.dialogue_pipeline._price_per_second_usd == pytest.approx(estimated_price_per_second_usd(0.098))
    assert app.conversation._price_per_second_usd == pytest.approx(estimated_price_per_second_usd(0.098))


def test_kling_is_off_by_default_in_settings_from_env(monkeypatch):
    monkeypatch.delenv("KLING_VIDEO_ENABLED", raising=False)
    monkeypatch.setenv("KLING_API_KEY", "k")
    assert Settings.from_env().kling_configured is False
