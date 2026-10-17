from __future__ import annotations

import pytest

from src.integrations.reasoning_client import MockAgentReasoningClient, RealGroqReasoningClient


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


def test_groq_complete_raises_clearly_on_a_real_error_response(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _FakeHttpxResponse({"error": {"message": "model not found"}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    client = RealGroqReasoningClient(api_key="fake-key", model="bad-model")

    with pytest.raises(RuntimeError, match="Groq completion failed"):
        client.complete("sys", "user")


def test_mock_agent_reasoning_client_scripts_replies_in_order():
    mock = MockAgentReasoningClient(scripted_replies=["first", "second"])

    assert mock.complete("s1", "u1") == "first"
    assert mock.complete("s2", "u2") == "second"
    assert mock.calls == [("s1", "u1"), ("s2", "u2")]
