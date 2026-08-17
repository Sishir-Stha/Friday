from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from friday.database.models import Conversation, Message


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