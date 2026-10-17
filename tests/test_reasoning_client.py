from __future__ import annotations

import pytest

from src.integrations.reasoning_client import MockAgentReasoningClient, RealGeminiReasoningClient, RealGroqReasoningClient


class _FakeHttpxResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_groq_complete_posts_messages_and_returns_content(monkeypatch):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json))
        return _FakeHttpxResponse({"choices": [{"message": {"content": '["a", "b"]'}}]})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealGroqReasoningClient(api_key="fake-key", model="openai/gpt-oss-120b")
    result = client.complete("system", "user")

    assert result == '["a", "b"]'
    url, headers, body = calls[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer fake-key"
    assert body["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]


def test_groq_complete_raises_clearly_on_a_real_error_response_with_no_fallback(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeHttpxResponse({"error": {"message": "model not found"}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealGroqReasoningClient(api_key="fake-key", model="bad-model")

    with pytest.raises(RuntimeError, match="Groq completion failed"):
        client.complete("sys", "user")


def test_groq_complete_falls_back_to_a_real_second_provider_on_error(monkeypatch):
    # Real finding 2026-10-17: a second Groq key under the same account
    # shares the same daily token quota - not real isolation. Gemini is a
    # genuinely separate account/quota, so Groq falls back to it.
    def fake_post(url, *, headers, json, timeout):
        raise RuntimeError(
            "Rate limit reached ... in organization org_01m4xav43be8j956tk3w9jt3tg ... tokens per day (TPD): Limit 200000, Used 199974"
        )

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    class _FakeFallback:
        def __init__(self):
            self.calls = []

        def complete(self, system_prompt, user_prompt):
            self.calls.append((system_prompt, user_prompt))
            return '["from gemini fallback"]'

    fallback = _FakeFallback()
    client = RealGroqReasoningClient(api_key="fake-key", model="openai/gpt-oss-120b", fallback=fallback)

    result = client.complete("sys", "user")

    assert result == '["from gemini fallback"]'
    assert fallback.calls == [("sys", "user")]


def test_groq_complete_with_no_fallback_still_raises_on_error(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        raise RuntimeError("boom")

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealGroqReasoningClient(api_key="fake-key", model="openai/gpt-oss-120b")

    with pytest.raises(RuntimeError, match="boom"):
        client.complete("sys", "user")


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


def _gemini_client(monkeypatch, response_text: str):
    calls = []

    class _FakeModels:
        def generate_content(self, *, model, contents):
            calls.append({"model": model, "contents": contents})
            return _FakeGeminiResponse(response_text)

    class _FakeClient:
        def __init__(self, *, api_key):
            self.models = _FakeModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", _FakeClient)

    client = RealGeminiReasoningClient(api_key="fake-key", model="gemma-4-26b-a4b-it")
    return client, calls


def test_gemini_complete_folds_the_system_prompt_into_contents(monkeypatch):
    client, calls = _gemini_client(monkeypatch, '["a", "b"]')

    result = client.complete("You are a writer.", "Write a screenplay.")

    assert len(calls) == 1
    assert calls[0] == {"model": "gemma-4-26b-a4b-it", "contents": "You are a writer.\n\nWrite a screenplay."}
    assert result == '["a", "b"]'


def test_gemini_complete_retries_a_transient_server_error_and_then_succeeds(monkeypatch):
    from google.genai import errors

    call_count = 0

    class _FakeModels:
        def generate_content(self, *, model, contents):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise errors.ServerError(500, {"error": {"message": "Internal error"}})
            return _FakeGeminiResponse('["ok"]')

    class _FakeClient:
        def __init__(self, *, api_key):
            self.models = _FakeModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    client = RealGeminiReasoningClient(api_key="fake-key", model="gemma-4-26b-a4b-it")
    result = client.complete("sys", "user")

    assert call_count == 3
    assert result == '["ok"]'


def test_gemini_complete_does_not_retry_a_non_rate_limit_client_error(monkeypatch):
    from google.genai import errors

    call_count = 0

    class _FakeModels:
        def generate_content(self, *, model, contents):
            nonlocal call_count
            call_count += 1
            raise errors.ClientError(400, {"error": {"message": "bad request"}})

    class _FakeClient:
        def __init__(self, *, api_key):
            self.models = _FakeModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setattr("time.sleep", lambda seconds: (_ for _ in ()).throw(AssertionError("should not sleep")))

    client = RealGeminiReasoningClient(api_key="fake-key", model="gemma-4-26b-a4b-it")

    with pytest.raises(errors.ClientError):
        client.complete("sys", "user")

    assert call_count == 1


def test_mock_agent_reasoning_client_scripts_replies_in_order():
    mock = MockAgentReasoningClient(scripted_replies=["first", "second"])

    assert mock.complete("s1", "u1") == "first"
    assert mock.complete("s2", "u2") == "second"
    assert mock.calls == [("s1", "u1"), ("s2", "u2")]
