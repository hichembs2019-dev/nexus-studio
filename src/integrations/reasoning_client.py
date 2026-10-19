"""Scene/screenplay planning - the one reasoning call the Writer and
Director agents each make.

The real Groq POST shape (https://api.groq.com/openai/v1/chat/completions,
Authorization: Bearer <key>, OpenAI-compatible choices[0].message.content)
is the same one verified live for nexus-commerce on 2026-10-14 (see that
project's docs/SOURCES.md "Groq fallback") - reused here as an
already-verified fact, not re-guessed.

Real finding, 2026-10-17 (see docs/SOURCES.md "Groq org-wide quota"): a
second Groq API key under the SAME Groq account/organization does NOT get
its own token-per-day quota - the real error body names the shared
`organization` id and the same used/limit numbers nexus-commerce's own
key was already hitting. A fresh key is not isolation; a different
provider is. RealGroqReasoningClient's optional `fallback` falls over to
RealGeminiReasoningClient (a genuinely separate account/quota) on any
Groq failure, same `fallback` pattern nexus-commerce already proved.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AgentReasoningClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """One request/response turn, returns the model's raw text reply."""


class RealGroqReasoningClient(AgentReasoningClient):
    def __init__(self, api_key: str, model: str, fallback: AgentReasoningClient | None = None):
        self._api_key = api_key
        self._model = model
        self._fallback = fallback

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        try:
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
            content = payload["choices"][0]["message"]["content"] or ""
            if not content.strip():
                # Real finding, 2026-10-19: a real 200 OK with a genuinely
                # empty choices[0].message.content (no "error" key at all)
                # slipped straight past the check above and was returned
                # as if it were a real reply - DialogueWriterAgent then
                # failed on "not valid JSON: ''" with no fallback ever
                # attempted, even though a real, separate-quota Gemini
                # fallback was configured and would have answered fine.
                # An empty completion is exactly as unusable as an error
                # one, so it is treated the same way here.
                raise RuntimeError("Groq completion failed: empty content in a successful response")
            return content
        except Exception:
            if self._fallback is not None:
                return self._fallback.complete(system_prompt, user_prompt)
            raise


class RealGeminiReasoningClient(AgentReasoningClient):
    """Same real shape verified live for nexus-commerce 2026-10-13/14 (see
    that project's docs/SOURCES.md "Gemini agent reasoning"): the system
    prompt is folded into `contents` rather than passed as
    `system_instruction` (that field 500s on at least the Gemma models
    nexus-commerce used), with a short retry for transient 500s and a
    longer one for a real 429 rate limit - reused here as already-verified
    facts, not re-guessed.
    """

    _MAX_RETRIES = 3
    _RETRY_DELAY_SECONDS = 2
    _RATE_LIMIT_RETRY_DELAY_SECONDS = 20

    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import errors

        contents = f"{system_prompt}\n\n{user_prompt}"
        last_error: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = self._client.models.generate_content(model=self._model, contents=contents)
                return response.text or ""
            except (errors.ServerError, errors.ClientError) as exc:
                is_rate_limit = getattr(exc, "code", None) == 429
                if not isinstance(exc, errors.ServerError) and not is_rate_limit:
                    raise
                last_error = exc
                if attempt < self._MAX_RETRIES - 1:
                    import time

                    time.sleep(self._RATE_LIMIT_RETRY_DELAY_SECONDS if is_rate_limit else self._RETRY_DELAY_SECONDS)
        raise last_error


class MockAgentReasoningClient(AgentReasoningClient):
    def __init__(self, scripted_replies: list[str] | None = None):
        self._replies = list(scripted_replies or [])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._replies:
            raise AssertionError("MockAgentReasoningClient ran out of scripted replies")
        return self._replies.pop(0)
