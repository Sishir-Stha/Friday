import pytest
from sqlalchemy import delete, select

from friday.database.connection import SessionLocal
from friday.database.models import Conversation, Message
from friday.llm.conversation_service import ConversationService

pytestmark = pytest.mark.integration


def test_conversation_service() -> None:
    service = ConversationService()

    conversation_id = service.create_conversation(
        title="Integration Test",
    )

    try:
        response = service.send_message(
            conversation_id,
            "Reply with exactly: Conversation pipeline working",
        )

        assert response.content.strip() == "Conversation pipeline working"

        with SessionLocal() as session:
            messages = list(
                session.scalars(
                    select(Message)
                    .where(
                        Message.conversation_id
                        == conversation_id
                    )
                    .order_by(Message.created_at)
                ).all()
            )

        assert len(messages) == 2

        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

        print("\nFriday response:")
        print(response.content)

    finally:
        with SessionLocal() as session:
            session.execute(
                delete(Conversation).where(
                    Conversation.id == conversation_id
                )
            )

            session.commit()
