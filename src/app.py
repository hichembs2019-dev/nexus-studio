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
from .integrations.fal_video_client import FalVideoClient, MockFalVideoClient, RealFalVideoClient
from .integrations.image_client import MockImageClient, RealFalImageClient
from .integrations.publishing_client import MockPublishingClient, PublishingClient, RealYouTubeClient
from .integrations.reasoning_client import (
    AgentReasoningClient,
    MockAgentReasoningClient,
    RealGeminiReasoningClient,
    RealGroqReasoningClient,
)
from .integrations.tts_client import MockTTSClient, RealAzureTTSClient, TTSClient
from .workflows.dialogue_pipeline import DialoguePipeline

logger = logging.getLogger("nexus_studio")


@dataclass
class App:
    settings: Settings
    reasoning: AgentReasoningClient
    fal_video: FalVideoClient | None
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
    # FAL_KEY/AZURE_SPEECH_KEY), since there is no fallback pipeline left
    # to drop back to for the bot at all.
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
    # fal.ai key (the generation itself) and a real Azure TTS key (the
    # audio each clip lip-syncs to) - a silent "dialogue" line defeats
    # the entire point, so this stays None rather than falling back to
    # a silent/mock pipeline.
    fal_video: FalVideoClient | None
    dialogue_writer: DialogueWriterAgent | None
    dialogue_pipeline: DialoguePipeline | None
    if not force_mocks and settings.fal_configured and settings.azure_speech_configured:
        fal_video = RealFalVideoClient(settings.fal_key)
        dialogue_writer = DialogueWriterAgent(reasoning)
        dialogue_pipeline = DialoguePipeline(
            fal_video, tts, settings.output_dir,
            tts_lang=settings.azure_tts_lang,
            max_cost_usd=settings.max_episode_cost_usd,
            # Owner's instruction, 2026-10-20: a real, story-specific
            # photo per character (not a fixed panel of six reused
            # forever) - same FAL_KEY, a second real endpoint
            # (fal-ai/flux/dev, see image_client.py).
            image_client=RealFalImageClient(settings.fal_key),
        )
    elif force_mocks:
        fal_video = MockFalVideoClient()
        dialogue_writer = DialogueWriterAgent(reasoning)
        dialogue_pipeline = DialoguePipeline(
            fal_video, tts or MockTTSClient(), settings.output_dir,
            tts_lang=settings.azure_tts_lang,
            image_client=MockImageClient(),
        )
    else:
        logger.warning("FAL_KEY and/or AZURE_SPEECH_KEY not set - real dialogue/lip-sync pipeline unavailable")
        fal_video = None
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
    conversation = BotConversation(dialogue_writer, publisher) if dialogue_writer is not None else None

    return App(
        settings=settings,
        reasoning=reasoning,
        fal_video=fal_video,
        dialogue_writer=dialogue_writer,
        dialogue_pipeline=dialogue_pipeline,
        publisher=publisher,
        conversation=conversation,
    )
