from dataclasses import dataclass
from typing import Any

import httpx

from friday.core.config import get_settings


@dataclass(frozen=True, slots=True)
class OllamaResponse:
    content: str
    thinking: str


class OllamaClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    f"{self.base_url}/api/tags",
                )

            return response.is_success

        except httpx.HTTPError:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> OllamaResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": True,
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=30.0,
            pool=10.0,
        )

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        message = data["message"]

        return OllamaResponse(
            content=message.get("content", "").strip(),
            thinking=message.get("thinking", "").strip(),
        )