from uuid import uuid4

import pytest

from friday.database.connection import SessionLocal
from friday.database.models import Memory
from friday.services.memory_service import MemoryService

pytestmark = pytest.mark.integration


def test_memory_service_postgresql_lifecycle() -> None:
    service = MemoryService()
    content = f"Friday memory integration smoke test {uuid4()}"
    memory_id: int | None = None

    try:
        created = service.remember(
            content,
            memory_type="integration_test",
            importance=7,
        )
        memory_id = created.id

        assert created.id > 0
        assert created.content == content
        assert created.memory_type == "integration_test"
        assert created.importance == 7
        assert created.is_active is True

        retrieved = service.get(memory_id)

        assert retrieved is not None
        assert retrieved.id == memory_id
        assert retrieved.content == content
        assert retrieved.memory_type == "integration_test"
        assert retrieved.importance == 7
        assert retrieved.is_active is True

        recalled = service.recall(
            limit=100,
            memory_type="integration_test",
        )
        assert memory_id in {memory.id for memory in recalled}

        assert service.forget(memory_id) is True

        inactive = service.get(memory_id)
        assert inactive is not None
        assert inactive.is_active is False

        recalled_after_forget = service.recall(
            limit=100,
            memory_type="integration_test",
        )
        assert memory_id not in {
            memory.id for memory in recalled_after_forget
        }

        assert service.restore(memory_id) is True

        restored = service.get(memory_id)
        assert restored is not None
        assert restored.is_active is True

        recalled_after_restore = service.recall(
            limit=100,
            memory_type="integration_test",
        )
        assert memory_id in {
            memory.id for memory in recalled_after_restore
        }
    finally:
        if memory_id is not None:
            with SessionLocal() as session:
                memory = session.get(Memory, memory_id)

                if memory is not None:
                    session.delete(memory)
                    session.commit()

            with SessionLocal() as session:
                assert session.get(Memory, memory_id) is None
