from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from friday.llm.conversation_service import ConversationService
from friday.llm.ollama_client import OllamaResponse, OllamaUnavailableError
from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.services.context_manager import ContextManager


class RecordingConversationService(ConversationService):
    def __init__(
        self,
        *,
        ollama: Any,
        state_machine: AssistantStateMachine,
        context_manager: ContextManager,
        failing_conversation_id: int | None = None,
    ) -> None:
        settings = SimpleNamespace(
            ai_mode="local",
            ollama_model="test-model",
        )
        super().__init__(
            ollama=ollama,
            settings=settings,
            state_machine=state_machine,
            context_manager=context_manager,
        )
        self.persisted_messages: list[tuple[str, str]] = []
        self.failing_conversation_id = failing_conversation_id

    def _persist_user_message(
        self,
        conversation_id: int,
        content: str,
    ) -> None:
        if conversation_id == self.failing_conversation_id:
            raise ValueError(f"Conversation {conversation_id} does not exist.")

        self.persisted_messages.append(("user", content))

    def _persist_assistant_message(
        self,
        conversation_id: int,
        content: str,
    ) -> None:
        self.persisted_messages.append(("assistant", content))

    def _build_llm_messages(
        self,
        conversation_id: int,
    ) -> list[dict[str, str]]:
        return [
            {"role": role, "content": content}
            for role, content in self.persisted_messages
        ]


class ContextAwareChatOllama:
    def __init__(
        self,
        machine: AssistantStateMachine,
        context: ContextManager,
        conversation_id: int,
    ) -> None:
        self.machine = machine
        self.context = context
        self.conversation_id = conversation_id

    def chat(self, messages: list[dict[str, str]]) -> OllamaResponse:
        assert self.machine.get() is AssistantState.PROCESSING
        assert (
            self.context.get().current_conversation_id
            == self.conversation_id
        )
        assert messages == [{"role": "user", "content": "Hello"}]
        return OllamaResponse(content="Hi")


class ContextAwareStreamingOllama:
    def __init__(
        self,
        machine: AssistantStateMachine,
        context: ContextManager,
        conversation_id: int,
    ) -> None:
        self.machine = machine
        self.context = context
        self.conversation_id = conversation_id

    def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        assert self.machine.get() is AssistantState.STREAMING
        assert (
            self.context.get().current_conversation_id
            == self.conversation_id
        )
        yield "Hello"
        assert (
            self.context.get().current_conversation_id
            == self.conversation_id
        )
        yield " world"


class UnavailableChatOllama:
    def chat(self, messages: list[dict[str, str]]) -> OllamaResponse:
        raise OllamaUnavailableError(
            "Friday cannot reach the local AI service. Make sure Ollama is running.",
            endpoint="http://127.0.0.1:11434/api/chat",
        )


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
    *,
    machine: AssistantStateMachine | None = None,
    context: ContextManager | None = None,
    failing_conversation_id: int | None = None,
) -> RecordingConversationService:
    return RecordingConversationService(
        ollama=ollama,
        state_machine=(
            machine if machine is not None else AssistantStateMachine()
        ),
        context_manager=context if context is not None else ContextManager(),
        failing_conversation_id=failing_conversation_id,
    )


def test_service_uses_injected_context_manager() -> None:
    context = ContextManager()

    service = make_service(object(), context=context)

    assert service.context_manager is context


def test_normal_chat_sets_and_retains_current_conversation() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    service = make_service(
        ContextAwareChatOllama(machine, context, 4),
        machine=machine,
        context=context,
    )

    response = service.send_message(4, "Hello")

    assert response.content == "Hi"
    assert machine.get() is AssistantState.IDLE
    assert context.get().current_conversation_id == 4
    assert service.persisted_messages == [
        ("user", "Hello"),
        ("assistant", "Hi"),
    ]


def test_stream_sets_context_before_return_and_retains_it() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    service = make_service(
        ContextAwareStreamingOllama(machine, context, 1),
        machine=machine,
        context=context,
    )

    stream = service.send_message_stream(1, "Hello")

    assert machine.get() is AssistantState.PROCESSING
    assert context.get().current_conversation_id == 1
    assert service.persisted_messages == [("user", "Hello")]

    assert list(stream) == ["Hello", " world"]
    assert machine.get() is AssistantState.IDLE
    assert context.get().current_conversation_id == 1
    assert service.persisted_messages == [
        ("user", "Hello"),
        ("assistant", "Hello world"),
    ]


def test_ollama_unavailable_retains_active_conversation() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    service = make_service(
        UnavailableChatOllama(),
        machine=machine,
        context=context,
    )

    with pytest.raises(OllamaUnavailableError):
        service.send_message(8, "Hello")

    assert machine.get() is AssistantState.OFFLINE
    assert context.get().current_conversation_id == 8
    assert service.persisted_messages == [("user", "Hello")]


def test_generic_error_retains_active_conversation() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    service = make_service(
        FailingChatOllama(),
        machine=machine,
        context=context,
    )

    with pytest.raises(RuntimeError, match="boom"):
        service.send_message(6, "Hello")

    assert machine.get() is AssistantState.ERROR
    assert context.get().current_conversation_id == 6
    assert service.persisted_messages == [("user", "Hello")]


def test_streaming_error_retains_active_conversation() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    service = make_service(
        FailingStreamingOllama(),
        machine=machine,
        context=context,
    )

    stream = service.send_message_stream(7, "Hello")
    with pytest.raises(RuntimeError, match="boom"):
        list(stream)

    assert machine.get() is AssistantState.ERROR
    assert context.get().current_conversation_id == 7
    assert service.persisted_messages == [("user", "Hello")]


def test_stream_cancellation_retains_current_conversation() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    service = make_service(
        CancellableStreamingOllama(),
        machine=machine,
        context=context,
    )
    stream = service.send_message_stream(3, "Hello")

    assert next(stream) == "partial"
    assert machine.get() is AssistantState.STREAMING
    assert context.get().current_conversation_id == 3

    stream.close()

    assert machine.get() is AssistantState.IDLE
    assert context.get().current_conversation_id == 3
    assert service.persisted_messages == [("user", "Hello")]


@pytest.mark.parametrize("method_name", ["send_message", "send_message_stream"])
def test_empty_message_preserves_existing_context(method_name: str) -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    context.update(current_conversation_id=2)
    service = make_service(
        object(),
        machine=machine,
        context=context,
    )
    method = getattr(service, method_name)

    with pytest.raises(ValueError, match="Message cannot be empty"):
        method(3, "   ")

    assert machine.get() is AssistantState.IDLE
    assert context.get().current_conversation_id == 2
    assert service.persisted_messages == []


def test_failed_user_persistence_does_not_replace_context() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    context.update(current_conversation_id=2)
    service = make_service(
        object(),
        machine=machine,
        context=context,
        failing_conversation_id=999,
    )

    with pytest.raises(ValueError, match="Conversation 999 does not exist"):
        service.send_message(999, "Hello")

    assert machine.get() is AssistantState.ERROR
    assert context.get().current_conversation_id == 2
    assert service.persisted_messages == []


def test_conversation_update_preserves_other_context_fields() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    context.update(
        active_module="system",
        active_window="Task Manager",
        selected_item="gpu",
        recent_tool_result="GPU usage 84%",
    )
    service = make_service(
        ContextAwareChatOllama(machine, context, 4),
        machine=machine,
        context=context,
    )

    service.send_message(4, "Hello")

    snapshot = context.get()
    assert snapshot.active_module == "system"
    assert snapshot.active_window == "Task Manager"
    assert snapshot.selected_item == "gpu"
    assert snapshot.recent_tool_result == "GPU usage 84%"
    assert snapshot.current_conversation_id == 4


def test_default_context_managers_are_independent() -> None:
    settings = SimpleNamespace(
        ai_mode="local",
        ollama_model="test-model",
    )
    first = ConversationService(ollama=object(), settings=settings)
    second = ConversationService(ollama=object(), settings=settings)

    first.context_manager.update(current_conversation_id=1)

    assert first.context_manager is not second.context_manager
    assert first.context_manager.get().current_conversation_id == 1
    assert second.context_manager.get().current_conversation_id is None
