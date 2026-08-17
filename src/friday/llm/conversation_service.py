from friday.core.config import get_settings
from friday.database.connection import SessionLocal
from friday.database.repositories import ConversationRepository
from friday.llm.ollama_client import OllamaClient, OllamaResponse
from friday.llm.prompts import FRIDAY_SYSTEM_PROMPT


class ConversationService:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.settings = get_settings()

    def create_conversation(
        self,
        title: str | None = None,
    ) -> int:
        with SessionLocal() as session:
            repository = ConversationRepository(session)

            conversation = repository.create_conversation(
                title=title,
                mode=self.settings.ai_mode,
            )

            session.commit()

            return conversation.id

    def send_message(
        self,
        conversation_id: int,
        content: str,
    ) -> OllamaResponse:
        content = content.strip()

        if not content:
            raise ValueError("Message cannot be empty.")

        with SessionLocal() as session:
            repository = ConversationRepository(session)

            conversation = repository.get_conversation(
                conversation_id,
            )

            if conversation is None:
                raise ValueError(
                    f"Conversation {conversation_id} does not exist."
                )

            repository.add_message(
                conversation_id=conversation_id,
                role="user",
                content=content,
            )

            session.commit()

        messages = self._build_llm_messages(
            conversation_id,
        )

        response = self.ollama.chat(messages)

        with SessionLocal() as session:
            repository = ConversationRepository(session)

            repository.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response.content,
                model=self.settings.ollama_model,
                metadata_json={
                    "thinking": response.thinking,
                },
            )

            session.commit()

        return response

    def _build_llm_messages(
        self,
        conversation_id: int,
    ) -> list[dict[str, str]]:
        with SessionLocal() as session:
            repository = ConversationRepository(session)

            history = repository.get_recent_messages(
                conversation_id=conversation_id,
                limit=20,
            )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": FRIDAY_SYSTEM_PROMPT,
            }
        ]

        for message in history:
            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        return messages