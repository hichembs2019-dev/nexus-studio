"""Wires every piece together from a single Settings, same convention as
nexus-commerce's src/app.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .agents.director import DirectorAgent
from .agents.publisher import PublishingAgent
from .agents.writer import WriterAgent
from .config import Settings
from .integrations.comfyui_client import ComfyUIClient, MockComfyUIClient, RealComfyUIClient
from .integrations.publishing_client import MockPublishingClient, PublishingClient, RealYouTubeClient
from .integrations.reasoning_client import (
    AgentReasoningClient,
    MockAgentReasoningClient,
    RealGeminiReasoningClient,
    RealGroqReasoningClient,
)
from .integrations.remote_exec_client import MockRemoteExecClient, RealJupyterRemoteExecClient, RemoteExecClient
from .workflows.episode_pipeline import EpisodePipeline

logger = logging.getLogger("nexus_studio")


@dataclass
class App:
    settings: Settings
    comfyui: ComfyUIClient
    remote_exec: RemoteExecClient | None
    reasoning: AgentReasoningClient
    writer: WriterAgent
    director: DirectorAgent
    pipeline: EpisodePipeline
    publisher: PublishingAgent


def build_app(settings: Settings, *, force_mocks: bool = False) -> App:
    if not force_mocks and settings.comfyui_configured:
        comfyui: ComfyUIClient = RealComfyUIClient(settings.comfyui_base_url)
    else:
        if not force_mocks:
            logger.warning("COMFYUI_BASE_URL not set - using MockComfyUIClient")
        comfyui = MockComfyUIClient()

    remote_exec: RemoteExecClient | None
    if not force_mocks and settings.jupyter_configured:
        remote_exec = RealJupyterRemoteExecClient(settings.jupyter_base_url, settings.jupyter_password)
    else:
        if not force_mocks:
            logger.warning("JUPYTER_BASE_URL/JUPYTER_PASSWORD not set - no remote stitching, single-scene episodes only")
        remote_exec = None if not force_mocks else MockRemoteExecClient()

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

    writer = WriterAgent(reasoning)
    director = DirectorAgent(reasoning)
    pipeline = EpisodePipeline(comfyui, settings.output_dir, remote_exec)

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

    return App(
        settings=settings,
        comfyui=comfyui,
        remote_exec=remote_exec,
        reasoning=reasoning,
        writer=writer,
        director=director,
        pipeline=pipeline,
        publisher=publisher,
    )
