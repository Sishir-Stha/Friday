from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from friday.llm.conversation_service import ConversationService
from friday.llm.ollama_client import OllamaResponse, OllamaUnavailableError
from friday.services.assistant_state import (
    AssistantState,
    AssistantStateMachine,
    InvalidStateTransition,
)


class RecordingConversationService(ConversationService):
    def __init__(
        self,
        *,
        ollama: Any,
        state_machine: AssistantStateMachine,
    ) -> None:
        settings = SimpleNamespace(
            ai_mode="local",
            ollama_model="test-model",
        )
        super().__init__(
            ollama=ollama,
            settings=settings,
            state_machine=state_machine,
        )
        self.persisted_messages: list[tuple[str, str]] = []
        self.assistant_persistence_states: list[AssistantState] = []

    def _persist_user_message(
        self,
        conversation_id: int,
        content: str,
    ) -> None:
        self.persisted_messages.append(("user", content))

    def _persist_assistant_message(
        self,
        conversation_id: int,
        content: str,
    ) -> None:
        self.assistant_persistence_states.append(self.state_machine.get())
        self.persisted_messages.append(("assistant", content))

    def _build_llm_messages(
        self,
        conversation_id: int,
    ) -> list[dict[str, str]]:
        return [
            {"role": role, "content": content}
            for role, content in self.persisted_messages
        ]


class SuccessfulChatOllama:
    def __init__(self, machine: AssistantStateMachine) -> None:
        self.machine = machine

    def chat(self, messages: list[dict[str, str]]) -> OllamaResponse:
        assert self.machine.get() is AssistantState.PROCESSING
        assert messages == [{"role": "user", "content": "Hello"}]
        return OllamaResponse(content="Hi")


class SuccessfulStreamingOllama:
    def __init__(self, machine: AssistantStateMachine) -> None:
        self.machine = machine

    def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        assert self.machine.get() is AssistantState.STREAMING
        yield "Hello"
        assert self.machine.get() is AssistantState.STREAMING
        yield " world"


class UnavailableChatOllama:
    def __init__(self) -> None:
        self.error = OllamaUnavailableError(
            "Friday cannot reach the local AI service. Make sure Ollama is running.",
            endpoint="http://127.0.0.1:11434/api/chat",
        )

    def chat(self, messages: list[dict[str, str]]) -> OllamaResponse:
        raise self.error


class UnavailableStreamingOllama:
    def __init__(self) -> None:
        self.error = OllamaUnavailableError(
            "Friday cannot reach the local AI service. Make sure Ollama is running.",
            endpoint="http://127.0.0.1:11434/api/chat",
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        yield "partial"
        raise self.error


class FailingChatOllama:
    def chat(self, messages: list[dict[str, str]]) -> OllamaResponse:
        raise RuntimeError("boom")


class FailingStreamingOllama:
    def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        yield "partial"
        raise RuntimeError("boom")


class CancellableStreamingOllama:
    def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        yield "partial"
        yield "unused"


def make_service(
    ollama: Any,
    machine: AssistantStateMachine | None = None,
) -> RecordingConversationService:
    return RecordingConversationService(
        ollama=ollama,
        state_machine=machine or AssistantStateMachine(),
    )


def test_new_service_starts_idle_and_uses_injected_machine() -> None:
    machine = AssistantStateMachine()

    service = make_service(SuccessfulChatOllama(machine), machine)

    assert service.state_machine is machine
    assert machine.get() is AssistantState.IDLE


def test_normal_chat_transitions_processing_to_idle() -> None:
    machine = AssistantStateMachine()
    service = make_service(SuccessfulChatOllama(machine), machine)

    response = service.send_message(1, "Hello")

    assert response.content == "Hi"
    assert machine.get() is AssistantState.IDLE
    assert service.persisted_messages == [
        ("user", "Hello"),
        ("assistant", "Hi"),
    ]
    assert service.assistant_persistence_states == [AssistantState.PROCESSING]


def test_stream_transitions_processing_streaming_idle() -> None:
    machine = AssistantStateMachine()
    service = make_service(SuccessfulStreamingOllama(machine), machine)

    stream = service.send_message_stream(1, "Hello")

    assert machine.get() is AssistantState.PROCESSING
    assert service.persisted_messages == [("user", "Hello")]
    assert list(stream) == ["Hello", " world"]
    assert machine.get() is AssistantState.IDLE
    assert service.persisted_messages == [
        ("user", "Hello"),
        ("assistant", "Hello world"),
    ]
    assert service.assistant_persistence_states == [AssistantState.STREAMING]


def test_normal_ollama_unavailable_transitions_offline() -> None:
    machine = AssistantStateMachine()
    ollama = UnavailableChatOllama()
    service = make_service(ollama, machine)

    with pytest.raises(OllamaUnavailableError) as exc_info:
        service.send_message(1, "Hello")

    assert exc_info.value is ollama.error
    assert machine.get() is AssistantState.OFFLINE
    assert service.persisted_messages == [("user", "Hello")]


def test_stream_ollama_unavailable_transitions_offline() -> None:
    machine = AssistantStateMachine()
    ollama = UnavailableStreamingOllama()
    service = make_service(ollama, machine)

    stream = service.send_message_stream(1, "Hello")
    with pytest.raises(OllamaUnavailableError) as exc_info:
        list(stream)

    assert exc_info.value is ollama.error
    assert machine.get() is AssistantState.OFFLINE
    assert service.persisted_messages == [("user", "Hello")]


def test_generic_normal_failure_transitions_error() -> None:
    machine = AssistantStateMachine()
    service = make_service(FailingChatOllama(), machine)

    with pytest.raises(RuntimeError, match="boom"):
        service.send_message(1, "Hello")

    assert machine.get() is AssistantState.ERROR
    assert service.persisted_messages == [("user", "Hello")]


def test_generic_streaming_failure_transitions_error() -> None:
    machine = AssistantStateMachine()
    service = make_service(FailingStreamingOllama(), machine)

    stream = service.send_message_stream(1, "Hello")
    with pytest.raises(RuntimeError, match="boom"):
        list(stream)

    assert machine.get() is AssistantState.ERROR
    assert service.persisted_messages == [("user", "Hello")]


def test_stream_cancellation_returns_idle_without_assistant_message() -> None:
    machine = AssistantStateMachine()
    service = make_service(CancellableStreamingOllama(), machine)
    stream = service.send_message_stream(1, "Hello")

    assert next(stream) == "partial"
    assert machine.get() is AssistantState.STREAMING

    stream.close()

    assert machine.get() is AssistantState.IDLE
    assert service.persisted_messages == [("user", "Hello")]


@pytest.mark.parametrize("method_name", ["send_message", "send_message_stream"])
def test_empty_message_does_not_change_state(method_name: str) -> None:
    machine = AssistantStateMachine()
    service = make_service(SuccessfulChatOllama(machine), machine)
    method = getattr(service, method_name)

    with pytest.raises(ValueError, match="Message cannot be empty"):
        method(1, "   ")

    assert machine.get() is AssistantState.IDLE
    assert service.persisted_messages == []


def test_initial_state_transition_refusal_is_not_classified_as_error() -> None:
    machine = AssistantStateMachine()
    machine.transition(AssistantState.OFFLINE)
    service = make_service(SuccessfulChatOllama(machine), machine)

    with pytest.raises(InvalidStateTransition):
        service.send_message(1, "Hello")

    assert machine.get() is AssistantState.OFFLINE
    assert service.persisted_messages == []
