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
            output_dir=os.environ.get("OUTPUT_DIR", "./output"),
        )
