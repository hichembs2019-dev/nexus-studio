"""Wires every piece together from a single Settings, same convention as
nexus-commerce's src/app.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .agents.dialogue_writer import DialogueWriterAgent
from .agents.publisher import PublishingAgent
from .bot.conversation import BotConversation
from .config import Settings
from .integrations.publishing_client import MockPublishingClient, PublishingClient, RealYouTubeClient
from .integrations.reasoning_client import (
    AgentReasoningClient,
    MockAgentReasoningClient,
    RealGeminiReasoningClient,
    RealGroqReasoningClient,
)
from .integrations.replicate_image_client import (
    PRICE_PER_IMAGE_USD_UNVERIFIED,
    MockReplicateImageClient,
    RealReplicateImageClient,
    ReplicateImageClient,
)
from .integrations.replicate_video_client import (
    PROVISIONAL_PRICE_PER_SECOND_USD,
    MockReplicateVideoClient,
    RealReplicateVideoClient,
    ReplicateVideoClient,
)
from .integrations.tts_client import MockTTSClient, RealAzureTTSClient, TTSClient
from .workflows.dialogue_pipeline import DialoguePipeline

logger = logging.getLogger("nexus_studio")


@dataclass
class App:
    settings: Settings
    reasoning: AgentReasoningClient
    video_client: ReplicateVideoClient | None
    dialogue_writer: DialogueWriterAgent | None
    dialogue_pipeline: DialoguePipeline | None
    publisher: PublishingAgent
    # Owner's standing instruction, 2026-10-21 ("احذف الراوي للابد الا
    # بطلب مني" - delete the narrator permanently unless I ask for it
    # myself): the old WriterAgent/DirectorAgent/EpisodePipeline/
    # ComfyUIClient/RemoteExecClient narration pipeline this App used to
    # also build (always, unconditionally, on every boot, real or mock)
    # is deleted outright - see models.py's own docstring. conversation
    # is None exactly when dialogue_writer/dialogue_pipeline are (no real
    # REPLICATE_API_TOKEN/AZURE_SPEECH_KEY), since there is no fallback
    # pipeline left to drop back to for the bot at all.
    conversation: BotConversation | None


def build_app(settings: Settings, *, force_mocks: bool = False) -> App:
    # Real finding 2026-10-17 (docs/SOURCES.md "Groq org-wide quota"): a
    # second Groq key under the same account shares nexus-commerce's own
    # daily token quota - not real isolation. Gemini is a genuinely
    # separate account/quota, so it rides along as Groq's fallback here,
    # same wiring nexus-commerce already uses (src/app.py there).
    if not force_mocks and settings.groq_configured:
        gemini_fallback = (
            RealGeminiReasoningClient(settings.gemini_api_key, settings.gemini_model)
            if settings.gemini_configured
            else None
        )
        reasoning: AgentReasoningClient = RealGroqReasoningClient(
            settings.groq_api_key, settings.groq_model, fallback=gemini_fallback
        )
    elif not force_mocks and settings.gemini_configured:
        reasoning = RealGeminiReasoningClient(settings.gemini_api_key, settings.gemini_model)
    else:
        if not force_mocks:
            logger.warning("Neither GROQ_API_KEY nor GEMINI_API_KEY set - using MockAgentReasoningClient")
        reasoning = MockAgentReasoningClient()

    tts: TTSClient | None
    if not force_mocks and settings.azure_speech_configured:
        tts = RealAzureTTSClient(settings.azure_speech_key, settings.azure_speech_region)
    else:
        if not force_mocks:
            logger.warning("AZURE_SPEECH_KEY/REGION not set - real dialogue/lip-sync pipeline unavailable")
        tts = MockTTSClient() if force_mocks else None

    # Real character dialogue with lip sync (DialoguePipeline) - the
    # owner's only real ask (see models.py's own docstring: no narrator-
    # style pipeline exists in this codebase at all). Needs BOTH a real
    # Replicate token (the generation itself) and a real Azure TTS key
    # (the audio each clip lip-syncs to) - a silent "dialogue" line
    # defeats the entire point, so this stays None rather than falling
    # back to a silent/mock pipeline.
    #
    # Owner's standing instruction, 2026-10-21 ("لا تضع السراق ...
    # احذفهم من الكود" - don't put the thieves [fal.ai] in, remove them
    # from the code, repeated explicitly a second time, "خذ وقتك ...
    # هذا ما طلبته منك بالظبط" - take your time, this is exactly what I
    # asked you): fal.ai is not merely unused by default any more - its
    # client code (fal_video_client.py, image_client.py) is deleted
    # outright from this repository, the same way the old narrator
    # pipeline and the unused telegram_client.py wrapper were. Replicate
    # (wan-video/wan-2.2-s2v for video, black-forest-labs/flux-dev for
    # images) is the only real video/image provider this app can build -
    # confirmed via a real, owner-run paid test whose quoted cost matched
    # its real billed cost exactly, unlike fal.ai's own real invoice (see
    # docs/SOURCES.md). There is no fallback provider to drop back to;
    # an unconfigured REPLICATE_API_TOKEN means no real pipeline at all,
    # same "no silent stand-in" posture this project already applies
    # everywhere else (character photos, no fallback face; etc).
    video_client: ReplicateVideoClient | None
    image_client: ReplicateImageClient | None
    dialogue_writer: DialogueWriterAgent | None
    dialogue_pipeline: DialoguePipeline | None
    # Declared here (not only inside the branch below) so BotConversation's
    # own pre-Approve estimate always has a real rate to use, even on a
    # mock/unconfigured build where no real pipeline exists at all.
    price_per_second_usd: float = PROVISIONAL_PRICE_PER_SECOND_USD
    price_per_image_usd: float = PRICE_PER_IMAGE_USD_UNVERIFIED
    if not force_mocks and settings.replicate_configured and settings.azure_speech_configured:
        video_client = RealReplicateVideoClient(settings.replicate_api_token)
        image_client = RealReplicateImageClient(settings.replicate_api_token)
        logger.info("Real video provider: Replicate (wan-video/wan-2.2-s2v)")
        logger.info("Real image provider: Replicate (black-forest-labs/flux-dev)")
        dialogue_writer = DialogueWriterAgent(reasoning)
        dialogue_pipeline = DialoguePipeline(
            video_client, tts, settings.output_dir,
            tts_lang=settings.azure_tts_lang,
            max_cost_usd=settings.max_episode_cost_usd,
            image_client=image_client,
            price_per_second_usd=price_per_second_usd,
            price_per_image_usd=price_per_image_usd,
        )
    elif force_mocks:
        video_client = MockReplicateVideoClient()
        dialogue_writer = DialogueWriterAgent(reasoning)
        dialogue_pipeline = DialoguePipeline(
            video_client, tts or MockTTSClient(), settings.output_dir,
            tts_lang=settings.azure_tts_lang,
            image_client=MockReplicateImageClient(),
            price_per_second_usd=price_per_second_usd,
            price_per_image_usd=price_per_image_usd,
        )
    else:
        logger.warning(
            "REPLICATE_API_TOKEN and/or AZURE_SPEECH_KEY not set - real dialogue/lip-sync pipeline unavailable"
        )
        video_client = None
        dialogue_writer = None
        dialogue_pipeline = None

    # Only YouTube has a Real client today (see docs/SOURCES.md "Publishing
    # - platforms not yet built" for what every other platform needs first).
    publishing_clients: dict[str, PublishingClient] = {}
    if not force_mocks and settings.youtube_configured:
        publishing_clients["youtube"] = RealYouTubeClient(
            settings.youtube_client_id, settings.youtube_client_secret, settings.youtube_refresh_token
        )
    else:
        if not force_mocks:
            logger.warning("YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN not set - 'youtube' has no publishing client")
        if force_mocks:
            publishing_clients["youtube"] = MockPublishingClient()
    publisher = PublishingAgent(publishing_clients)
    conversation = (
        BotConversation(
            dialogue_writer, publisher,
            price_per_second_usd=price_per_second_usd, price_per_image_usd=price_per_image_usd,
        )
        if dialogue_writer is not None
        else None
    )

    return App(
        settings=settings,
        reasoning=reasoning,
        video_client=video_client,
        dialogue_writer=dialogue_writer,
        dialogue_pipeline=dialogue_pipeline,
        publisher=publisher,
        conversation=conversation,
    )
