from uuid import uuid4

import pytest
from sqlalchemy import select

from friday.database.connection import SessionLocal
from friday.database.models import Conversation, Memory, Message
from friday.llm.conversation_service import ConversationService
from friday.services.assistant_state import AssistantState
from friday.services.memory_service import MemoryService

pytestmark = pytest.mark.integration


def test_postgresql_memory_reaches_real_ollama_response() -> None:
    memory_service = MemoryService()
    conversation_service = ConversationService()
    memory_value = f"FRIDAYMEM{uuid4().hex[:12].upper()}"
    question = (
        "What is my integration memory code? "
        "Reply with only the code and nothing else."
    )
    memory_id: int | None = None
    conversation_id: int | None = None

    assert memory_value not in question

    try:
        memory = memory_service.remember(
            f"The user's integration memory code is {memory_value}.",
            memory_type="integration_test",
            importance=10,
        )
        memory_id = memory.id

        conversation_id = conversation_service.create_conversation(
            title="Memory Ollama Integration Test",
        )

        response = conversation_service.send_message(
            conversation_id,
            question,
        )

        print("\n===== FRIDAY MEMORY ANSWER =====")
        print(response.content)

        answer = response.content.strip()
        assert answer == memory_value
        assert conversation_service.state_machine.get() is AssistantState.IDLE
        assert (
            conversation_service.context_manager.get().current_conversation_id
            == conversation_id
        )

        with SessionLocal() as session:
            messages = list(
                session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at, Message.id)
                ).all()
            )
            stored_memory = session.get(Memory, memory_id)

        assert len(messages) == 2
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].content == question
        assert messages[1].content.strip() == memory_value
        assert stored_memory is not None
        assert stored_memory.is_active is True
    finally:
        with SessionLocal() as session:
            if memory_id is not None:
                stored_memory = session.get(Memory, memory_id)

                if stored_memory is not None:
                    session.delete(stored_memory)

            if conversation_id is not None:
                conversation = session.get(Conversation, conversation_id)

                if conversation is not None:
                    session.delete(conversation)

            session.commit()

        with SessionLocal() as session:
            if memory_id is not None:
                assert session.get(Memory, memory_id) is None

            if conversation_id is not None:
                assert session.get(Conversation, conversation_id) is None
                remaining_message_ids = list(
                    session.scalars(
                        select(Message.id).where(
                            Message.conversation_id == conversation_id
                        )
                    ).all()
                )
                assert remaining_message_ids == []
