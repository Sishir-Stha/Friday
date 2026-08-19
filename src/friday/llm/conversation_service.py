from collections.abc import Iterator
from typing import Protocol

from friday.core.config import get_settings
from friday.database.connection import SessionLocal
from friday.database.repositories import ConversationRepository
from friday.llm.ollama_client import OllamaClient, OllamaResponse
from friday.llm.prompts import FRIDAY_SYSTEM_PROMPT


class ConversationSettings(Protocol):
    ai_mode: str
    ollama_model: str


class ConversationService:
    def __init__(
        self,
        *,
        ollama: OllamaClient | None = None,
        settings: ConversationSettings | None = None,
    ) -> None:
        self.ollama = ollama if ollama is not None else OllamaClient()
        self.settings = settings if settings is not None else get_settings()

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
        content = self._validate_content(content)
        self._persist_user_message(conversation_id, content)

        messages = self._build_llm_messages(
            conversation_id,
        )

        response = self.ollama.chat(messages)

        self._persist_assistant_message(
            conversation_id,
            response.content,
        )

        return response

    def send_message_stream(
        self,
        conversation_id: int,
        content: str,
    ) -> Iterator[str]:
        """Return an iterator of chunks and persist one completed response."""
        content = self._validate_content(content)
        self._persist_user_message(conversation_id, content)

        messages = self._build_llm_messages(conversation_id)

        return self._stream_and_persist(
            conversation_id,
            messages,
        )

    def _stream_and_persist(
        self,
        conversation_id: int,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        chunks: list[str] = []

        for chunk in self.ollama.chat_stream(messages):
            if not chunk:
                continue

            chunks.append(chunk)
            yield chunk

        complete_content = "".join(chunks)
        if not complete_content.strip():
            raise ValueError("Ollama completed without assistant content.")

        self._persist_assistant_message(
            conversation_id,
            complete_content,
        )

    @staticmethod
    def _validate_content(content: str) -> str:
        content = content.strip()

        if not content:
            raise ValueError("Message cannot be empty.")

        return content

    @staticmethod
    def _persist_user_message(
        conversation_id: int,
        content: str,
    ) -> None:
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

    def _persist_assistant_message(
        self,
        conversation_id: int,
        content: str,
    ) -> None:
        with SessionLocal() as session:
            repository = ConversationRepository(session)

            repository.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                model=self.settings.ollama_model,
            )

            session.commit()

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
