import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NoReturn

import httpx

from friday.core.config import get_settings

ChatMessage = dict[str, Any]


@dataclass(frozen=True, slots=True)
class OllamaToolCall:
    name: str
    arguments: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool call name must be a non-empty string.")
        if not isinstance(self.arguments, dict) or any(
            not isinstance(key, str) for key in self.arguments
        ):
            raise ValueError("Tool call arguments must be an object with string keys.")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True)
class OllamaResponse:
    content: str
    tool_calls: tuple[OllamaToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class OllamaHealth:
    version: str


class OllamaError(Exception):
    """Base exception for failures while communicating with Ollama."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.detail = detail


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached."""


class OllamaTimeoutError(OllamaError):
    """Ollama did not respond within the configured timeout."""


class OllamaResponseError(OllamaError):
    """Ollama returned an HTTP or response-format error."""


class OllamaClient:
    HEALTH_TIMEOUT_SECONDS = 2.0
    CHAT_TIMEOUT = httpx.Timeout(
        connect=10.0,
        read=600.0,
        write=30.0,
        pool=10.0,
    )

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if base_url is None or model is None:
            settings = get_settings()
            base_url = base_url or settings.ollama_base_url
            model = model or settings.ollama_model

        self.base_url = base_url.rstrip("/")
        self.model = model
        self._transport = transport

    def health_check(self) -> OllamaHealth:
        """Check the Ollama server without loading an AI model."""
        endpoint = f"{self.base_url}/api/version"

        try:
            with self._create_client(
                timeout=self.HEALTH_TIMEOUT_SECONDS,
            ) as client:
                response = client.get(endpoint)
                self._ensure_success(response, endpoint)
                data = self._response_json(response, endpoint)
        except httpx.TimeoutException as exc:
            self._raise_timeout(exc, endpoint)
        except httpx.RequestError as exc:
            self._raise_unavailable(exc, endpoint)

        version = data.get("version")
        if not isinstance(version, str) or not version.strip():
            raise OllamaResponseError(
                "The local AI service returned an invalid health response.",
                endpoint=endpoint,
                detail="Response did not contain a non-empty 'version' string.",
            )

        return OllamaHealth(version=version.strip())

    def is_available(self) -> bool:
        """Backward-compatible boolean availability check."""
        try:
            self.health_check()
        except OllamaError:
            return False

        return True

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, object]] | None = None,
    ) -> OllamaResponse:
        endpoint = f"{self.base_url}/api/chat"
        payload = self._chat_payload(
            messages,
            stream=False,
            tools=tools,
        )

        try:
            with self._create_client(timeout=self.CHAT_TIMEOUT) as client:
                response = client.post(endpoint, json=payload)
                self._ensure_success(response, endpoint)
                data = self._response_json(response, endpoint)
        except httpx.TimeoutException as exc:
            self._raise_timeout(exc, endpoint)
        except httpx.RequestError as exc:
            self._raise_unavailable(exc, endpoint)

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaResponseError(
                "The local AI service returned an invalid response.",
                endpoint=endpoint,
                detail="Response did not contain a 'message' object.",
            )

        tool_calls = self._parse_tool_calls(
            message,
            endpoint=endpoint,
        )
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaResponseError(
                "The local AI service returned no assistant content.",
                endpoint=endpoint,
                detail="Assistant message content was missing or not a string.",
            )

        content = self._strip_embedded_thinking(content)
        if not content and not tool_calls:
            raise OllamaResponseError(
                "The local AI service returned no assistant content.",
                endpoint=endpoint,
                detail="Assistant message contained no final content.",
            )

        return OllamaResponse(
            content=content,
            tool_calls=tool_calls,
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
    ) -> Iterator[str]:
        """Yield visible assistant content from a streaming Ollama response."""
        endpoint = f"{self.base_url}/api/chat"
        payload = self._chat_payload(messages, stream=True)
        received_visible_content = False

        try:
            with (
                self._create_client(timeout=self.CHAT_TIMEOUT) as client,
                client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                ) as response,
            ):
                self._ensure_success(response, endpoint)

                for line_number, line in enumerate(
                    response.iter_lines(),
                    start=1,
                ):
                    if not line.strip():
                        continue

                    data = self._stream_json(
                        line,
                        endpoint=endpoint,
                        line_number=line_number,
                    )

                    error = data.get("error")
                    if error:
                        raise OllamaResponseError(
                            "The local AI service returned a streaming error.",
                            endpoint=endpoint,
                            detail=str(error)[:1000],
                        )

                    done = data.get("done")
                    if not isinstance(done, bool):
                        raise OllamaResponseError(
                            "The local AI service returned a malformed stream chunk.",
                            endpoint=endpoint,
                            detail=(
                                f"Stream line {line_number} did not contain a "
                                "boolean 'done' value."
                            ),
                        )

                    message = data.get("message")
                    if message is not None and not isinstance(message, dict):
                        raise OllamaResponseError(
                            "The local AI service returned a malformed stream chunk.",
                            endpoint=endpoint,
                            detail=(
                                f"Stream line {line_number} had a non-object "
                                "'message' value."
                            ),
                        )

                    if message is None and not done:
                        raise OllamaResponseError(
                            "The local AI service returned a malformed stream chunk.",
                            endpoint=endpoint,
                            detail=(
                                f"Stream line {line_number} did not contain a "
                                "'message' object."
                            ),
                        )

                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if not isinstance(content, str):
                            raise OllamaResponseError(
                                "The local AI service returned a malformed stream chunk.",
                                endpoint=endpoint,
                                detail=(
                                    f"Stream line {line_number} had non-string "
                                    "assistant content."
                                ),
                            )

                        if content:
                            if content.strip():
                                received_visible_content = True
                            yield content

                    if done:
                        if not received_visible_content:
                            raise OllamaResponseError(
                                "The local AI service returned no assistant content.",
                                endpoint=endpoint,
                                detail="Stream completed without assistant content.",
                            )
                        return

                raise OllamaResponseError(
                    "The local AI service ended the response unexpectedly.",
                    endpoint=endpoint,
                    detail="Streaming response ended without a completion marker.",
                )
        except httpx.TimeoutException as exc:
            self._raise_timeout(exc, endpoint)
        except httpx.RequestError as exc:
            self._raise_unavailable(exc, endpoint)

    def _create_client(
        self,
        *,
        timeout: float | httpx.Timeout,
    ) -> httpx.Client:
        return httpx.Client(
            timeout=timeout,
            transport=self._transport,
        )

    def _chat_payload(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool,
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,
        }
        if tools:
            payload["tools"] = tools
        return payload

    @staticmethod
    def _parse_tool_calls(
        message: dict[str, Any],
        *,
        endpoint: str,
    ) -> tuple[OllamaToolCall, ...]:
        if "tool_calls" not in message:
            return ()

        raw_calls = message["tool_calls"]
        if not isinstance(raw_calls, list):
            raise OllamaResponseError(
                "The local AI service returned invalid tool calls.",
                endpoint=endpoint,
                detail="Assistant 'tool_calls' was not a list.",
            )

        parsed_calls: list[OllamaToolCall] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                OllamaClient._raise_invalid_tool_call(
                    endpoint,
                    index,
                    "tool call was not an object",
                )

            function = raw_call.get("function")
            if not isinstance(function, dict):
                OllamaClient._raise_invalid_tool_call(
                    endpoint,
                    index,
                    "function was not an object",
                )

            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                OllamaClient._raise_invalid_tool_call(
                    endpoint,
                    index,
                    "function name was not a non-empty string",
                )

            arguments = function.get("arguments")
            if not isinstance(arguments, dict):
                OllamaClient._raise_invalid_tool_call(
                    endpoint,
                    index,
                    "function arguments was not an object",
                )
            if any(not isinstance(key, str) for key in arguments):
                OllamaClient._raise_invalid_tool_call(
                    endpoint,
                    index,
                    "function arguments contained a non-string key",
                )

            parsed_calls.append(
                OllamaToolCall(
                    name=name,
                    arguments=arguments,
                )
            )

        return tuple(parsed_calls)

    @staticmethod
    def _raise_invalid_tool_call(
        endpoint: str,
        index: int,
        detail: str,
    ) -> NoReturn:
        raise OllamaResponseError(
            "The local AI service returned an invalid tool call.",
            endpoint=endpoint,
            detail=f"Tool call {index}: {detail}.",
        )

    @staticmethod
    def _ensure_success(
        response: httpx.Response,
        endpoint: str,
    ) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                "The local AI service returned an HTTP error.",
                endpoint=endpoint,
                status_code=response.status_code,
                detail=OllamaClient._response_detail(response),
            ) from exc

    @staticmethod
    def _response_json(
        response: httpx.Response,
        endpoint: str,
    ) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "The local AI service returned malformed JSON.",
                endpoint=endpoint,
                status_code=response.status_code,
                detail="Response body was not valid JSON.",
            ) from exc

        if not isinstance(data, dict):
            raise OllamaResponseError(
                "The local AI service returned an invalid response.",
                endpoint=endpoint,
                status_code=response.status_code,
                detail="Expected a JSON object.",
            )

        return data

    @staticmethod
    def _stream_json(
        line: str,
        *,
        endpoint: str,
        line_number: int,
    ) -> dict[str, Any]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(
                "The local AI service returned a malformed stream chunk.",
                endpoint=endpoint,
                detail=f"Invalid JSON on stream line {line_number}.",
            ) from exc

        if not isinstance(data, dict):
            raise OllamaResponseError(
                "The local AI service returned a malformed stream chunk.",
                endpoint=endpoint,
                detail=f"Stream line {line_number} was not a JSON object.",
            )

        return data

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            return response.text[:1000]
        except httpx.ResponseNotRead:
            response.read()
            return response.text[:1000]

    @staticmethod
    def _strip_embedded_thinking(content: str) -> str:
        """Defensively remove a model-template thinking preamble from content."""
        normalized = content.lower()
        closing_tag = "</think>"
        closing_index = normalized.rfind(closing_tag)

        if closing_index >= 0:
            content = content[closing_index + len(closing_tag) :]

        return content.strip()

    @staticmethod
    def _raise_timeout(
        exc: httpx.TimeoutException,
        endpoint: str,
    ) -> NoReturn:
        raise OllamaTimeoutError(
            "The local AI service did not respond in time.",
            endpoint=endpoint,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    @staticmethod
    def _raise_unavailable(
        exc: httpx.RequestError,
        endpoint: str,
    ) -> NoReturn:
        raise OllamaUnavailableError(
            "Friday cannot reach the local AI service. Make sure Ollama is running.",
            endpoint=endpoint,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
