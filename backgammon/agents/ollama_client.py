"""Minimal client for a local Ollama server (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("BACKGAMMON_MODEL", "qwen3.8:27b")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL, timeout: float = 180.0) -> None:
        if not host.startswith("http"):
            host = f"http://{host}"
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._request("/api/tags", timeout=5).get("models", [])]

    def is_available(self) -> bool:
        try:
            return self.model in self.list_models()
        except OllamaError:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
        temperature: float = 0.4,
        think: bool | None = False,
    ) -> str:
        """Send a chat and return the assistant's text. ``schema`` requests structured JSON output."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if schema is not None:
            payload["format"] = schema
        if think is not None:
            payload["think"] = think
        try:
            reply = self._request("/api/chat", payload)
        except OllamaError as exc:
            # Models without a thinking mode reject the "think" flag; retry without it.
            if think is not None and "think" in str(exc):
                payload.pop("think")
                reply = self._request("/api/chat", payload)
            else:
                raise
        return reply.get("message", {}).get("content", "")

    def chat_json(self, messages: list[dict[str, str]], schema: dict, **kwargs: Any) -> dict:
        text = self.chat(messages, schema=schema, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # Some models wrap JSON in prose or code fences despite the schema.
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise OllamaError(f"model did not return JSON: {text[:200]!r}") from exc
