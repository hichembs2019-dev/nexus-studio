"""Wires every piece together from a single Settings, same convention as
nexus-commerce's src/app.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .agents.director import DirectorAgent
from .agents.writer import WriterAgent
from .config import Settings
from .integrations.comfyui_client import ComfyUIClient, MockComfyUIClient, RealComfyUIClient
from .integrations.reasoning_client import AgentReasoningClient, MockAgentReasoningClient, RealGroqReasoningClient
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

    if not force_mocks and settings.groq_configured:
        reasoning: AgentReasoningClient = RealGroqReasoningClient(settings.groq_api_key, settings.groq_model)
    else:
        if not force_mocks:
            logger.warning("GROQ_API_KEY not set - using MockAgentReasoningClient")
        reasoning = MockAgentReasoningClient()

    writer = WriterAgent(reasoning)
    director = DirectorAgent(reasoning)
    pipeline = EpisodePipeline(comfyui, settings.output_dir, remote_exec)

    return App(
        settings=settings,
        comfyui=comfyui,
        remote_exec=remote_exec,
        reasoning=reasoning,
        writer=writer,
        director=director,
        pipeline=pipeline,
    )
