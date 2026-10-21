from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEGENERATE_ABORT = "token repeat limit reached"

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _ollama_root(base_url: str) -> str:
    root = str(base_url).rstrip("/")
    return root[:-3].rstrip("/") if root.endswith("/v1") else root


def _native_chat(base_url: str, model: str, messages: list[dict], temperature: float,
                 max_tokens: int, timeout: float, think: bool) -> str:
    body = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        _ollama_root(base_url) + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            payload = json.load(fh)
    except urllib.error.HTTPError as e:
        if e.code == 500 and DEGENERATE_ABORT in e.read().decode("utf-8", "replace"):
            return ""
        raise
    message = payload.get("message") or {}
    return message.get("content") or ""


def get_client(cfg: Any) -> Any:
    api_base = getattr(cfg.llm, "api_base", None) or "http://localhost:11434/v1"
    api_key = "ollama" if "11434" in api_base or "localhost" in api_base else os.environ.get("OPENAI_API_KEY", "")
    if OpenAI is None:
        raise ImportError("Install openai>=1.0: pip install openai")
    return OpenAI(api_key=api_key, base_url=api_base)


def call_llm(client: Any, model: str, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 1024, timeout: float = 120.0,
             think: bool | None = None) -> str:
    import re as _re
    for attempt in range(2):
        try:
            if think is not None:
                content = _native_chat(client.base_url, model, messages, temperature,
                                       max_tokens, timeout, think)
                return _re.sub(r"<think>.*?</think>", "", content,
                               flags=_re.DOTALL).strip()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            if not content.strip():
                finished = resp.choices[0].finish_reason == "stop"
                extra = getattr(msg, "model_extra", None) or {}
                reasoning = (getattr(msg, "reasoning_content", None)
                             or extra.get("reasoning")
                             or extra.get("reasoning_content")
                             or "")
                if finished and reasoning.strip():
                    content = reasoning
            content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                raise RuntimeError(f"LLM call failed after 2 attempts: {e}") from e
    return ""


def call_self_consistency(client: Any, model: str, prompt_chains: list[list[dict]],
                          temperature: float = 0.7, max_tokens: int = 1024,
                          timeout: float = 120.0, think: bool | None = None) -> list[str]:
    return [
        call_llm(client, model, msgs, temperature=temperature, max_tokens=max_tokens,
                 timeout=timeout, think=think)
        for msgs in prompt_chains
    ]
