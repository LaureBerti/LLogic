"""Thin LLM client wrapping OpenAI-compatible API (Ollama, OpenAI, Anthropic via proxy)."""
from __future__ import annotations
import os
import time
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def get_client(cfg: Any) -> Any:
    """Return an OpenAI-compatible client. Defaults to local Ollama."""
    api_base = getattr(cfg.llm, "api_base", None) or "http://localhost:11434/v1"
    api_key = "ollama" if "11434" in api_base or "localhost" in api_base else os.environ.get("OPENAI_API_KEY", "")
    if OpenAI is None:
        raise ImportError("Install openai>=1.0: pip install openai")
    return OpenAI(api_key=api_key, base_url=api_base)


def call_llm(client: Any, model: str, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 1024, timeout: float = 120.0) -> str:
    """Call LLM and return the response text. Retries once on timeout.

    Handles DeepSeek-R1 style models that may return thinking tokens in
    reasoning_content (Ollama) or embed them in content as <think>…</think>.
    Falls back to reasoning_content when content is empty.
    """
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            # DeepSeek-R1 via Ollama sometimes returns the answer in reasoning_content
            # when max_tokens is tight and thinking fills the budget.
            if not content.strip():
                reasoning = getattr(msg, "reasoning_content", None) or ""
                if reasoning.strip():
                    content = reasoning
            # Strip <think>…</think> blocks if still present in content
            import re as _re
            content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                raise RuntimeError(f"LLM call failed after 2 attempts: {e}") from e
    return ""  # unreachable


def call_self_consistency(client: Any, model: str, prompt_chains: list[list[dict]],
                          temperature: float = 0.7, max_tokens: int = 1024,
                          timeout: float = 120.0) -> list[str]:
    """Call LLM n times and return all responses for majority voting."""
    return [
        call_llm(client, model, msgs, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        for msgs in prompt_chains
    ]
