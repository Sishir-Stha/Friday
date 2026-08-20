from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from friday.database.models import Conversation, Memory, Message


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_conversation(
        self,
        title: str | None = None,
        mode: str = "local",
    ) -> Conversation:
        conversation = Conversation(
            title=title,
            mode=mode,
        )

        self.session.add(conversation)
        self.session.flush()

        return conversation

    def get_conversation(
        self,
        conversation_id: int,
    ) -> Conversation | None:
        return self.session.get(
            Conversation,
            conversation_id,
        )

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        model: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            metadata_json=metadata_json or {},
        )

        self.session.add(message)
        self.session.flush()

        return message

    def get_recent_messages(
        self,
        conversation_id: int,
        limit: int = 20,
    ) -> list[Message]:
        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )

        messages = list(
            self.session.scalars(statement).all()
        )

        messages.reverse()

        return messages


class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_memory(
        self,
        *,
        content: str,
        memory_type: str = "fact",
        importance: int = 5,
        source_message_id: int | None = None,
    ) -> Memory:
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            source_message_id=source_message_id,
            is_active=True,
        )

        self.session.add(memory)
        self.session.flush()

        return memory

    def get_memory(self, memory_id: int) -> Memory | None:
        return self.session.get(Memory, memory_id)

    def list_active_memories(
        self,
        *,
        limit: int = 20,
        memory_type: str | None = None,
    ) -> list[Memory]:
        statement = select(Memory).where(
            Memory.is_active.is_(True),
        )

        if memory_type is not None:
            statement = statement.where(
                Memory.memory_type == memory_type,
            )

        statement = statement.order_by(
            Memory.importance.desc(),
            Memory.updated_at.desc(),
            Memory.id.desc(),
        ).limit(limit)

        return list(self.session.scalars(statement).all())

    def deactivate_memory(
        self,
        memory_id: int,
    ) -> Memory | None:
        memory = self.get_memory(memory_id)

        if memory is None:
            return None

        memory.is_active = False
        self.session.flush()

        return memory

    def reactivate_memory(
        self,
        memory_id: int,
    ) -> Memory | None:
        memory = self.get_memory(memory_id)

        if memory is None:
            return None

        memory.is_active = True
        self.session.flush()

        return memory
