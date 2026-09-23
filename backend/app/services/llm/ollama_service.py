from typing import Any

import httpx

from app.core.config import get_settings


class OllamaService:
    """
    Local LLM service using Ollama.

    Responsible only for communicating with the
    Ollama HTTP API.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()

        self.base_url = (
            base_url or settings.ollama_base_url
        ).rstrip("/")

        self.model = (
            model or settings.ollama_model
        )

        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a response using the configured Ollama model.
        """

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to communicate with Ollama: {exc}"
            ) from exc

        data = response.json()

        generated_text = data.get("response")

        if not generated_text:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return generated_text.strip()
