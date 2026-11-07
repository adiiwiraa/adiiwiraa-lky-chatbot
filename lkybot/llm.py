"""OpenAI-compatible LLM wrapper.

Configuration comes from environment variables through lkybot.config:
LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL.
"""
from __future__ import annotations

import time

import requests

from lkybot import config


class LLMError(RuntimeError):
    """Raised when the LLM API cannot produce a response."""


def _require_key() -> str:
    if not config.LLM_API_KEY:
        raise LLMError(
            "LLM_API_KEY is not set. Export it in your shell or copy .env.example to .env."
        )
    return config.LLM_API_KEY


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> str:
    """Send a chat completion request and return the assistant text."""
    key = _require_key()
    if not config.LLM_BASE_URL:
        raise LLMError(
            "LLM_BASE_URL is not set. Export it in your shell or copy .env.example to .env."
        )
    url = f"{config.LLM_BASE_URL}/chat/completions"
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "lkybot/0.1",
    }

    last_error = ""
    for attempt in range(1, 4):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(attempt * 3)
                continue
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise LLMError("LLM returned an empty response.")
            return content.strip()
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_error = str(exc)
            if attempt < 3:
                time.sleep(attempt * 3)
                continue
    raise LLMError(f"LLM request failed: {last_error}")
