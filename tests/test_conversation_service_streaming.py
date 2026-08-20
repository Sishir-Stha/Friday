import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Self

import httpx
import pytest

import friday.llm.conversation_service as conversation_service_module
from friday.llm.conversation_service import ConversationService
from friday.llm.ollama_client import OllamaClient, OllamaUnavailableError


@dataclass
class FakeState:
    conversations: set[int] = field(default_factory=lambda: {1})
    messages: list[SimpleNamespace] = field(default_factory=list)


class FakeSession:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def commit(self) -> None:
        pass


class FakeConversationRepository:
    state: FakeState

    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def get_conversation(self, conversation_id: int) -> object | None:
        if conversation_id in self.state.conversations:
            return object()
        return None

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        model: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        message = SimpleNamespace(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            metadata_json=metadata_json or {},
        )
        self.state.messages.append(message)
        return message

    def get_recent_messages(
        self,
        conversation_id: int,
        limit: int = 20,
    ) -> list[SimpleNamespace]:
        return [
            message
            for message in self.state.messages[-limit:]
            if message.conversation_id == conversation_id
        ]


class StreamingOllama:
    def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        assert messages[-1] == {"role": "user", "content": "Hello"}
        yield "Hello"
        yield " "
        yield "world"


class FailingStreamingOllama:
    def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        yield "partial"
        raise OllamaUnavailableError(
            "Friday cannot reach the local AI service. Make sure Ollama is running.",
            endpoint="http://127.0.0.1:11434/api/chat",
        )


class EmptyMemoryService:
    def recall(
        self,
        *,
        limit: int = 20,
        memory_type: str | None = None,
    ) -> list[object]:
        return []


@pytest.fixture
def fake_state(monkeypatch: pytest.MonkeyPatch) -> FakeState:
    state = FakeState()
    FakeConversationRepository.state = state
    monkeypatch.setattr(
        conversation_service_module,
        "SessionLocal",
        FakeSession,
    )
    monkeypatch.setattr(
        conversation_service_module,
        "ConversationRepository",
        FakeConversationRepository,
    )
    return state


def make_service(ollama: Any) -> ConversationService:
    settings = SimpleNamespace(
        ai_mode="local",
        ollama_model="test-model",
    )
    return ConversationService(
        ollama=ollama,
        settings=settings,
        memory_service=EmptyMemoryService(),  # type: ignore[arg-type]
    )


def test_normal_chat_ignores_and_never_persists_thinking(
    fake_state: FakeState,
) -> None:
    historical_message = SimpleNamespace(
        conversation_id=1,
        role="assistant",
        content="Earlier answer",
        model="test-model",
        metadata_json={"thinking": "historical reasoning must stay private"},
    )
    fake_state.messages.append(historical_message)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["think"] is False
        assert "historical reasoning" not in json.dumps(payload["messages"])
        return httpx.Response(
            200,
            json={
                "message": {
                    "thinking": "long internal reasoning that must be ignored",
                    "content": "Hello",
                }
            },
        )

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    service = make_service(client)

    response = service.send_message(1, "Say hello")

    assert response.content == "Hello"
    assert [message.content for message in fake_state.messages] == [
        "Earlier answer",
        "Say hello",
        "Hello",
    ]
    assert fake_state.messages[-1].metadata_json == {}
    assert historical_message.metadata_json == {
        "thinking": "historical reasoning must stay private"
    }


def test_stream_persists_user_and_one_complete_assistant_message(
    fake_state: FakeState,
) -> None:
    service = make_service(StreamingOllama())

    stream = service.send_message_stream(1, " Hello ")

    assert [message.role for message in fake_state.messages] == ["user"]
    assert list(stream) == ["Hello", " ", "world"]
    assert [message.role for message in fake_state.messages] == [
        "user",
        "assistant",
    ]

    assistant_messages = [
        message
        for message in fake_state.messages
        if message.role == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "Hello world"
    assert assistant_messages[0].model == "test-model"
    assert assistant_messages[0].metadata_json == {}


def test_partial_stream_failure_does_not_persist_assistant_message(
    fake_state: FakeState,
) -> None:
    service = make_service(FailingStreamingOllama())

    stream = service.send_message_stream(1, "Hello")

    with pytest.raises(OllamaUnavailableError):
        list(stream)

    assert [message.role for message in fake_state.messages] == ["user"]
