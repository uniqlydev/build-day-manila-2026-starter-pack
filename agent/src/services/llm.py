"""Shared OpenRouter client helpers for agent LLM calls."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from typing import Any

import httpx
from PIL import Image

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "google/gemini-2.5-pro-preview"


class OpenRouterClient:
    """Minimal async OpenRouter client for vision+text chat calls."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("TEAM_TOKEN", "")
        self.model = model or os.getenv("LLM_API_KEY", _DEFAULT_MODEL)
        self.timeout_s = timeout_s

        if not self.api_key:
            raise ValueError(
                "Missing TEAM_TOKEN in environment; required for OpenRouter auth."
            )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    async def run_vision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image,
        temperature: float = 0.2,
    ) -> str:
        image_b64 = self._encode_image(image)

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(_OPENROUTER_URL, headers=headers, json=payload)

        response.raise_for_status()
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from exc


def try_parse_json(content: str) -> dict[str, Any]:
    """Parse JSON from either raw JSON or fenced markdown content."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)
