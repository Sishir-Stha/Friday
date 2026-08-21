from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select

from friday.database.connection import SessionLocal
from friday.database.models import Conversation, Message
from friday.llm.conversation_service import ConversationService
from friday.llm.ollama_client import ChatMessage, OllamaClient, OllamaResponse
from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.services.context_manager import ContextManager
from friday.tools.builtins import ToolRuntime
from friday.tools.executor import ToolExecutor
from friday.tools.models import ToolDefinition, ToolRisk
from friday.tools.registry import ToolRegistry

pytestmark = pytest.mark.integration


class RecordingOllamaClient(OllamaClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[
            tuple[list[ChatMessage], list[dict[str, object]] | None]
        ] = []
        self.responses: list[OllamaResponse] = []

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, object]] | None = None,
    ) -> OllamaResponse:
        self.calls.append((deepcopy(messages), deepcopy(tools)))
        response = super().chat(messages, tools=tools)
        self.responses.append(response)
        return response


class EmptyMemoryService:
    def recall(self, *, limit: int) -> list[Any]:
        return []


def test_real_ollama_executes_deterministic_read_only_tool() -> None:
    token = f"FRIDAYTOOL{uuid4().hex[:12].upper()}"
    question = (
        "What is the integration verification token? Use the available tool "
        "and reply with only the token."
    )
    assert token not in question

    machine = AssistantStateMachine()
    context = ContextManager()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_integration_token",
            description=(
                "Return the current temporary integration verification token. "
                "Use this tool whenever the user asks for the integration "
                "verification token."
            ),
            category="integration_test",
            risk=ToolRisk.READ_ONLY,
        )
    )
    executor = ToolExecutor(registry=registry, state_machine=machine)
    handler_calls: list[dict[str, object]] = []

    def get_integration_token(arguments: dict[str, object]) -> dict[str, str]:
        handler_calls.append(arguments)
        return {"token": token}

    executor.bind("get_integration_token", get_integration_token)
    runtime = ToolRuntime(registry=registry, executor=executor)
    ollama = RecordingOllamaClient()
    service = ConversationService(
        ollama=ollama,
        settings=SimpleNamespace(ai_mode="local", ollama_model=ollama.model),
        state_machine=machine,
        context_manager=context,
        memory_service=EmptyMemoryService(),  # type: ignore[arg-type]
        tool_runtime=runtime,
    )
    conversation_id: int | None = None

    try:
        conversation_id = service.create_conversation(
            title="Native Tool Ollama Integration Test"
        )
        response = service.send_message(conversation_id, question)

        first_response, second_response = ollama.responses
        tool_messages = [
            message
            for message in ollama.calls[1][0]
            if message["role"] == "tool"
        ]

        print("\n===== FIRST OLLAMA RESPONSE =====")
        print(f"content={first_response.content!r}")
        print(f"tool_calls={first_response.tool_calls!r}")
        print("\n===== EXECUTED TOOL RESULT =====")
        print(tool_messages[-1]["content"])
        print("\n===== SECOND OLLAMA RESPONSE =====")
        print(f"content={second_response.content!r}")
        print(f"tool_calls={second_response.tool_calls!r}")

        assert response.content.strip() == token
        assert handler_calls == [{}]
        assert machine.get() is AssistantState.IDLE
        snapshot = context.get()
        assert snapshot.current_conversation_id == conversation_id
        assert snapshot.recent_tool_result is not None
        assert "get_integration_token" in snapshot.recent_tool_result
        assert token in snapshot.recent_tool_result

        with SessionLocal() as session:
            messages = list(
                session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at, Message.id)
                ).all()
            )

        assert len(messages) == 2
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].content == question
        assert token not in messages[0].content
        assert messages[1].content.strip() == token
    finally:
        if conversation_id is not None:
            with SessionLocal() as session:
                conversation = session.get(Conversation, conversation_id)
                if conversation is not None:
                    session.delete(conversation)
                session.commit()

            with SessionLocal() as session:
                assert session.get(Conversation, conversation_id) is None
                remaining_message_ids = list(
                    session.scalars(
                        select(Message.id).where(
                            Message.conversation_id == conversation_id
                        )
                    ).all()
                )
                assert remaining_message_ids == []
                remaining_test_conversations = list(
                    session.scalars(
                        select(Conversation.id).where(
                            Conversation.title
                            == "Native Tool Ollama Integration Test"
                        )
                    ).all()
                )
                print(
                    "remaining_test_conversations="
                    f"{len(remaining_test_conversations)}"
                )
