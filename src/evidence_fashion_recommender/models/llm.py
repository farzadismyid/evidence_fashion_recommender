"""Local Ollama generation adapter."""

from __future__ import annotations

import json
import urllib.request

from ..config import LLMConfig


class OllamaGenerator:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model_id = (
            f"{config.name}@{config.expected_digest or 'unversioned'}"
            f":temperature={config.temperature}:max_tokens={config.max_tokens}"
            f":think={config.think}"
        )

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.config.name,
                "prompt": prompt,
                "stream": False,
                "think": self.config.think,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.endpoint.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))["response"].strip()
