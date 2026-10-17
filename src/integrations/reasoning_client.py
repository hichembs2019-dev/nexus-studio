"""Scene planning - the one reasoning call the Director Agent makes to
turn a premise into N scene prompts.

The real POST shape (https://api.groq.com/openai/v1/chat/completions,
Authorization: Bearer <key>, OpenAI-compatible choices[0].message.content)
is the same one verified live for nexus-commerce on 2026-10-14 (see that
project's docs/SOURCES.md "Groq fallback") - reused here as an
already-verified fact, not re-guessed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AgentReasoningClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """One request/response turn, returns the model's raw text reply."""


class RealGroqReasoningClient(AgentReasoningClient):
    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=60,
        )
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"Groq completion failed: {payload['error'].get('message')}")
        return payload["choices"][0]["message"]["content"] or ""


class MockAgentReasoningClient(AgentReasoningClient):
    def __init__(self, scripted_replies: list[str] | None = None):
        self._replies = list(scripted_replies or [])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._replies:
            raise AssertionError("MockAgentReasoningClient ran out of scripted replies")
        return self._replies.pop(0)
