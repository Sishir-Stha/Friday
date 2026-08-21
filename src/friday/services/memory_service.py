from dataclasses import dataclass

from friday.database.connection import SessionLocal
from friday.database.models import Memory
from friday.database.repositories import MemoryRepository


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    id: int
    memory_type: str
    content: str
    importance: int
    source_message_id: int | None
    is_active: bool


class MemoryService:
    def remember(
        self,
        content: str,
        *,
        memory_type: str = "fact",
        importance: int = 5,
        source_message_id: int | None = None,
    ) -> MemorySnapshot:
        normalized_content = self._normalize_required_string(
            content,
            field_name="content",
        )
        normalized_memory_type = self._normalize_required_string(
            memory_type,
            field_name="memory_type",
        )
        self._validate_bounded_integer(
            importance,
            field_name="importance",
            minimum=1,
            maximum=10,
        )
        self._validate_optional_positive_integer(
            source_message_id,
            field_name="source_message_id",
        )

        with SessionLocal() as session:
            repository = MemoryRepository(session)
            memory = repository.create_memory(
                content=normalized_content,
                memory_type=normalized_memory_type,
                importance=importance,
                source_message_id=source_message_id,
            )
            session.commit()

            return _memory_to_snapshot(memory)

    def recall(
        self,
        *,
        limit: int = 20,
        memory_type: str | None = None,
    ) -> list[MemorySnapshot]:
        self._validate_bounded_integer(
            limit,
            field_name="limit",
            minimum=1,
            maximum=100,
        )
        normalized_memory_type = (
            None
            if memory_type is None
            else self._normalize_required_string(
                memory_type,
                field_name="memory_type",
            )
        )

        with SessionLocal() as session:
            repository = MemoryRepository(session)
            memories = repository.list_active_memories(
                limit=limit,
                memory_type=normalized_memory_type,
            )

            return [_memory_to_snapshot(memory) for memory in memories]

    def forget(self, memory_id: int) -> bool:
        self._validate_positive_integer(
            memory_id,
            field_name="memory_id",
        )

        with SessionLocal() as session:
            repository = MemoryRepository(session)
            memory = repository.deactivate_memory(memory_id)

            if memory is None:
                return False

            session.commit()
            return True

    def restore(self, memory_id: int) -> bool:
        self._validate_positive_integer(
            memory_id,
            field_name="memory_id",
        )

        with SessionLocal() as session:
            repository = MemoryRepository(session)
            memory = repository.reactivate_memory(memory_id)

            if memory is None:
                return False

            session.commit()
            return True

    def get(self, memory_id: int) -> MemorySnapshot | None:
        self._validate_positive_integer(
            memory_id,
            field_name="memory_id",
        )

        with SessionLocal() as session:
            repository = MemoryRepository(session)
            memory = repository.get_memory(memory_id)

            if memory is None:
                return None

            return _memory_to_snapshot(memory)

    @staticmethod
    def _normalize_required_string(value: str, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(  # noqa: TRY004
                f"{field_name} must be a non-empty string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{field_name} must be a non-empty string")

        return normalized

    @classmethod
    def _validate_optional_positive_integer(
        cls,
        value: int | None,
        *,
        field_name: str,
    ) -> None:
        if value is not None:
            cls._validate_positive_integer(value, field_name=field_name)

    @staticmethod
    def _validate_positive_integer(value: int, *, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    @staticmethod
    def _validate_bounded_integer(
        value: int,
        *,
        field_name: str,
        minimum: int,
        maximum: int,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"{field_name} must be an integer between {minimum} and {maximum}"
            )


def _memory_to_snapshot(memory: Memory) -> MemorySnapshot:
    return MemorySnapshot(
        id=memory.id,
        memory_type=memory.memory_type,
        content=memory.content,
        importance=memory.importance,
        source_message_id=memory.source_message_id,
        is_active=memory.is_active,
    )
