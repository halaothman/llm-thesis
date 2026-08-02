from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from openai import APIStatusError, OpenAI, RateLimitError

from .config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_REASONER_MAX_TOKENS,
    DEFAULT_DEEPSEEK_MODEL,
    LLM_INSUFFICIENT_BALANCE,
    LLM_LIMIT_ERROR,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_REQUEST_TOO_LARGE,
)

Provider = str

DEEPSEEK_REASONER_MODELS = frozenset({"deepseek-reasoner", "deepseek-r1"})


def _is_deepseek_reasoner(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered in DEEPSEEK_REASONER_MODELS or "reasoner" in lowered


DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"


def deepseek_balance_status(api_key: str | None) -> dict:
    if not api_key:
        return {"ok": False, "error": "no_key"}

    try:
        request = urllib.request.Request(
            DEEPSEEK_BALANCE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            parsed = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}

    balance_infos = parsed.get("balance_infos") or []
    usd = next(
        (item for item in balance_infos if str(item.get("currency", "")).upper() == "USD"),
        balance_infos[0] if balance_infos else {},
    )
    total = str(usd.get("total_balance", "")).strip()
    topped_up = str(usd.get("topped_up_balance", "")).strip()
    currency = str(usd.get("currency", "USD")).upper() or "USD"

    return {
        "ok": True,
        "is_available": bool(parsed.get("is_available")),
        "currency": currency,
        "total_balance": total,
        "topped_up_balance": topped_up,
    }


def create_llm_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
    )


def chat_complete(
    provider: Provider,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    temperature: float = 0.4,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    del provider
    resolved_model = model or DEFAULT_DEEPSEEK_MODEL
    reasoner = _is_deepseek_reasoner(resolved_model)
    request_kwargs: dict = {
        "model": resolved_model,
        "messages": messages,
    }
    if not reasoner:
        request_kwargs["temperature"] = temperature
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    elif reasoner:
        request_kwargs["max_tokens"] = DEEPSEEK_REASONER_MAX_TOKENS
    elif json_mode:
        request_kwargs["max_tokens"] = LLM_MAX_COMPLETION_TOKENS
    use_json_mode = json_mode and not reasoner
    if use_json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}

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
