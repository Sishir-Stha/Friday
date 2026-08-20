import json

import httpx
import pytest

from friday.llm.ollama_client import (
    OllamaClient,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

BASE_URL = "http://127.0.0.1:11434"
MESSAGES = [{"role": "user", "content": "Hello"}]


def make_client(
    handler: httpx.MockTransport,
    *,
    model: str = "test-model",
) -> OllamaClient:
    return OllamaClient(
        base_url=BASE_URL,
        model=model,
        transport=handler,
    )


def test_health_check_succeeds_without_loading_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/version"
        assert not request.content
        return httpx.Response(200, json={"version": "0.12.0"})

    client = make_client(httpx.MockTransport(handler))

    health = client.health_check()

    assert health.version == "0.12.0"
    assert client.is_available()


def test_chat_connection_refused_uses_domain_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaUnavailableError) as exc_info:
        client.chat(MESSAGES)

    assert "Make sure Ollama is running" in str(exc_info.value)
    assert exc_info.value.endpoint == f"{BASE_URL}/api/chat"
    assert "ConnectError" in (exc_info.value.detail or "")


@pytest.mark.parametrize(
    "timeout_error",
    [httpx.ConnectTimeout, httpx.ReadTimeout],
)
def test_chat_timeout_uses_domain_exception(
    timeout_error: type[httpx.TimeoutException],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise timeout_error("Timed out", request=request)

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaTimeoutError) as exc_info:
        client.chat(MESSAGES)

    assert timeout_error.__name__ in (exc_info.value.detail or "")


def test_chat_http_error_uses_response_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Ollama is unavailable")

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaResponseError) as exc_info:
        client.chat(MESSAGES)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Ollama is unavailable"


def test_chat_rejects_malformed_response_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaResponseError, match="malformed JSON"):
        client.chat(MESSAGES)


def test_chat_rejects_missing_assistant_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"thinking": "ignored"}})

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaResponseError, match="no assistant content"):
        client.chat(MESSAGES)


def test_chat_parses_successful_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert payload["stream"] is False
        assert payload["think"] is False
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": " Friday is ready. ",
                    "thinking": "long internal reasoning that must be ignored",
                }
            },
        )

    client = make_client(httpx.MockTransport(handler))

    response = client.chat(MESSAGES)

    assert response.content == "Friday is ready."
    assert not hasattr(response, "thinking")


def test_chat_removes_embedded_thinking_preamble() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": (
                        "internal reasoning that must be ignored\n"
                        "</think>\n\nHello"
                    )
                }
            },
        )

    client = make_client(httpx.MockTransport(handler))

    response = client.chat(MESSAGES)

    assert response.content == "Hello"


def test_chat_stream_combines_chunks_and_ignores_empty_content() -> None:
    chunks = [
        {
            "message": {
                "thinking": "long internal reasoning that must be ignored",
                "content": "Hello",
            },
            "done": False,
        },
        {"message": {"thinking": "still ignored"}, "done": False},
        {"message": {"content": ""}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["think"] is False
        content = "\n".join(json.dumps(chunk) for chunk in chunks)
        return httpx.Response(200, content=content)

    client = make_client(httpx.MockTransport(handler))

    streamed_chunks = list(client.chat_stream(MESSAGES))

    assert streamed_chunks == ["Hello", " world"]
    assert "".join(streamed_chunks) == "Hello world"
    assert all("reasoning" not in chunk for chunk in streamed_chunks)
    assert all("ignored" not in chunk for chunk in streamed_chunks)


def test_chat_stream_connection_failure_uses_domain_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaUnavailableError):
        list(client.chat_stream(MESSAGES))


def test_chat_stream_rejects_malformed_chunk() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content='{"message":')

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaResponseError, match="malformed stream chunk"):
        list(client.chat_stream(MESSAGES))


def test_chat_stream_http_error_uses_response_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content="stream failed")

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaResponseError) as exc_info:
        list(client.chat_stream(MESSAGES))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "stream failed"


def test_chat_stream_requires_completion_marker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(
                {"message": {"content": "partial"}, "done": False}
            ),
        )

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaResponseError, match="ended.*unexpectedly"):
        list(client.chat_stream(MESSAGES))
