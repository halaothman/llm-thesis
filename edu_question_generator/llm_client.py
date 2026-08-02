"""عميل DeepSeek R1 API: محادثة."""
from __future__ import annotations

import os

from openai import APIStatusError, OpenAI, RateLimitError

from .config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_R1_MAX_TOKENS,
    DEEPSEEK_R1_MODEL,
    LLM_INSUFFICIENT_BALANCE,
    LLM_LIMIT_ERROR,
    LLM_REQUEST_TOO_LARGE,
)


def create_llm_client(api_key: str | None = None) -> OpenAI:
    """إنشاء عميل OpenAI-compatible متجه إلى DeepSeek."""
    return OpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
    )


def chat_complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """استدعاء DeepSeek R1 — بدون temperature ولا json_mode."""
    request_kwargs: dict = {
        "model": model or DEEPSEEK_R1_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else DEEPSEEK_R1_MAX_TOKENS,
    }

    client = create_llm_client(api_key)
    try:
        response = client.chat.completions.create(**request_kwargs)
    except RateLimitError as exc:
        raise RuntimeError(LLM_LIMIT_ERROR) from exc
    except APIStatusError as exc:
        if exc.status_code == 413:
            raise RuntimeError(LLM_REQUEST_TOO_LARGE) from exc
        if exc.status_code == 429:
            raise RuntimeError(LLM_LIMIT_ERROR) from exc
        if exc.status_code == 402:
            raise RuntimeError(LLM_INSUFFICIENT_BALANCE) from exc
        raise
    return response.choices[0].message.content or "{}"
