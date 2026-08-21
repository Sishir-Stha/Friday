from dataclasses import dataclass, field
from typing import Any, Self

import pytest
from sqlalchemy.dialects import postgresql

import friday.services.memory_service as memory_service_module
from friday.database.repositories import MemoryRepository
from friday.services.memory_service import MemoryService, MemorySnapshot


@dataclass
class FakeMemory:
    id: int
    memory_type: str
    content: str
    importance: int
    source_message_id: int | None
    is_active: bool
    updated_order: int


@dataclass
class FakeStore:
    memories: dict[int, FakeMemory] = field(default_factory=dict)
    next_id: int = 1
    update_counter: int = 0
    commits: int = 0
    sessions_opened: int = 0
    last_recall: tuple[int, str | None] | None = None

    def next_update_order(self) -> int:
        self.update_counter += 1
        return self.update_counter


class FakeSession:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def commit(self) -> None:
        self.store.commits += 1


class FakeSessionFactory:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def __call__(self) -> FakeSession:
        self.store.sessions_opened += 1
        return FakeSession(self.store)


class FakeMemoryRepository:
    def __init__(self, session: FakeSession) -> None:
        self.store = session.store

    def create_memory(
        self,
        *,
        content: str,
        memory_type: str = "fact",
        importance: int = 5,
        source_message_id: int | None = None,
    ) -> FakeMemory:
        memory = FakeMemory(
            id=self.store.next_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            source_message_id=source_message_id,
            is_active=True,
            updated_order=self.store.next_update_order(),
        )
        self.store.memories[memory.id] = memory
        self.store.next_id += 1
        return memory

    def get_memory(self, memory_id: int) -> FakeMemory | None:
        return self.store.memories.get(memory_id)

    def list_active_memories(
        self,
        *,
        limit: int = 20,
        memory_type: str | None = None,
    ) -> list[FakeMemory]:
        self.store.last_recall = (limit, memory_type)
        memories = [
            memory
            for memory in self.store.memories.values()
            if memory.is_active
            and (memory_type is None or memory.memory_type == memory_type)
        ]
        memories.sort(
            key=lambda memory: (
                memory.importance,
                memory.updated_order,
                memory.id,
            ),
            reverse=True,
        )
        return memories[:limit]

    def deactivate_memory(self, memory_id: int) -> FakeMemory | None:
        memory = self.get_memory(memory_id)
        if memory is None:
            return None
        memory.is_active = False
        memory.updated_order = self.store.next_update_order()
        return memory

    def reactivate_memory(self, memory_id: int) -> FakeMemory | None:
        memory = self.get_memory(memory_id)
        if memory is None:
            return None
        memory.is_active = True
        memory.updated_order = self.store.next_update_order()
        return memory


@pytest.fixture
def memory_store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    store = FakeStore()
    monkeypatch.setattr(
        memory_service_module,
        "SessionLocal",
        FakeSessionFactory(store),
    )
    monkeypatch.setattr(
        memory_service_module,
        "MemoryRepository",
        FakeMemoryRepository,
    )
    return store


def test_remember_normalizes_values_and_returns_snapshot(
    memory_store: FakeStore,
) -> None:
    snapshot = MemoryService().remember(
        "  User prefers dark mode  ",
        memory_type="  preference  ",
        importance=8,
    )

    assert snapshot == MemorySnapshot(
        id=1,
        memory_type="preference",
        content="User prefers dark mode",
        importance=8,
        source_message_id=None,
        is_active=True,
    )
    assert memory_store.commits == 1


@pytest.mark.parametrize("content", ["", "   "])
def test_remember_rejects_empty_content(
    memory_store: FakeStore,
    content: str,
) -> None:
    with pytest.raises(ValueError, match="content"):
        MemoryService().remember(content)

    assert memory_store.sessions_opened == 0


@pytest.mark.parametrize("memory_type", ["", "   "])
def test_remember_rejects_empty_memory_type(
    memory_store: FakeStore,
    memory_type: str,
) -> None:
    with pytest.raises(ValueError, match="memory_type"):
        MemoryService().remember("A fact", memory_type=memory_type)

    assert memory_store.sessions_opened == 0


@pytest.mark.parametrize("importance", [1, 10])
def test_remember_accepts_importance_boundaries(
    memory_store: FakeStore,
    importance: int,
) -> None:
    snapshot = MemoryService().remember("A fact", importance=importance)

    assert snapshot.importance == importance


@pytest.mark.parametrize("importance", [0, 11, -1, True, False, 1.5])
def test_remember_rejects_invalid_importance(
    memory_store: FakeStore,
    importance: Any,
) -> None:
    with pytest.raises(ValueError, match="importance"):
        MemoryService().remember("A fact", importance=importance)

    assert memory_store.sessions_opened == 0


@pytest.mark.parametrize("source_message_id", [None, 1, 100])
def test_remember_accepts_source_message_id(
    memory_store: FakeStore,
    source_message_id: int | None,
) -> None:
    snapshot = MemoryService().remember(
        "A fact",
        source_message_id=source_message_id,
    )

    assert snapshot.source_message_id == source_message_id


@pytest.mark.parametrize("source_message_id", [0, -1, True, False])
def test_remember_rejects_invalid_source_message_id(
    memory_store: FakeStore,
    source_message_id: Any,
) -> None:
    with pytest.raises(ValueError, match="source_message_id"):
        MemoryService().remember(
            "A fact",
            source_message_id=source_message_id,
        )

    assert memory_store.sessions_opened == 0


def test_recall_returns_only_active_memories(memory_store: FakeStore) -> None:
    service = MemoryService()
    active = service.remember("Active")
    inactive = service.remember("Inactive")
    assert service.forget(inactive.id) is True

    recalled = service.recall()

    assert [memory.id for memory in recalled] == [active.id]


def test_recall_normalizes_and_filters_memory_type(
    memory_store: FakeStore,
) -> None:
    service = MemoryService()
    preference = service.remember("Likes tea", memory_type="preference")
    service.remember("Lives locally", memory_type="fact")

    recalled = service.recall(memory_type="  preference  ")

    assert [memory.id for memory in recalled] == [preference.id]
    assert memory_store.last_recall == (20, "preference")


@pytest.mark.parametrize("memory_type", ["", "   "])
def test_recall_rejects_empty_memory_type(
    memory_store: FakeStore,
    memory_type: str,
) -> None:
    with pytest.raises(ValueError, match="memory_type"):
        MemoryService().recall(memory_type=memory_type)

    assert memory_store.sessions_opened == 0


def test_recall_orders_by_importance_then_update_then_id(
    memory_store: FakeStore,
) -> None:
    service = MemoryService()
    low = service.remember("Low", importance=2)
    equal_older = service.remember("Equal older", importance=7)
    equal_newer = service.remember("Equal newer", importance=7)
    high = service.remember("High", importance=10)

    recalled = service.recall()

    assert [memory.id for memory in recalled] == [
        high.id,
        equal_newer.id,
        equal_older.id,
        low.id,
    ]


@pytest.mark.parametrize("limit", [1, 20, 100])
def test_recall_accepts_valid_limits(
    memory_store: FakeStore,
    limit: int,
) -> None:
    MemoryService().recall(limit=limit)

    assert memory_store.last_recall == (limit, None)


@pytest.mark.parametrize("limit", [0, -1, 101, True])
def test_recall_rejects_invalid_limits(
    memory_store: FakeStore,
    limit: Any,
) -> None:
    with pytest.raises(ValueError, match="limit"):
        MemoryService().recall(limit=limit)

    assert memory_store.sessions_opened == 0


def test_forget_soft_deactivates_existing_memory(
    memory_store: FakeStore,
) -> None:
    service = MemoryService()
    snapshot = service.remember("Remember me")

    assert service.forget(snapshot.id) is True
    assert memory_store.memories[snapshot.id].is_active is False
    assert service.recall() == []
    assert memory_store.commits == 2


def test_forget_nonexistent_memory_returns_false(
    memory_store: FakeStore,
) -> None:
    assert MemoryService().forget(999) is False
    assert memory_store.commits == 0


def test_restore_reactivates_memory(memory_store: FakeStore) -> None:
    service = MemoryService()
    snapshot = service.remember("Restore me")
    service.forget(snapshot.id)

    assert service.restore(snapshot.id) is True
    assert memory_store.memories[snapshot.id].is_active is True
    assert [memory.id for memory in service.recall()] == [snapshot.id]
    assert memory_store.commits == 3


def test_restore_nonexistent_memory_returns_false(
    memory_store: FakeStore,
) -> None:
    assert MemoryService().restore(999) is False
    assert memory_store.commits == 0


def test_get_returns_inactive_memory(memory_store: FakeStore) -> None:
    service = MemoryService()
    snapshot = service.remember("Inactive but retrievable")
    service.forget(snapshot.id)

    retrieved = service.get(snapshot.id)

    assert retrieved is not None
    assert retrieved.id == snapshot.id
    assert retrieved.is_active is False


def test_get_nonexistent_memory_returns_none(memory_store: FakeStore) -> None:
    assert MemoryService().get(999) is None


@pytest.mark.parametrize("method_name", ["forget", "restore", "get"])
@pytest.mark.parametrize("memory_id", [0, -1, True, False, 1.5])
def test_memory_id_operations_reject_invalid_ids(
    memory_store: FakeStore,
    method_name: str,
    memory_id: Any,
) -> None:
    method = getattr(MemoryService(), method_name)

    with pytest.raises(ValueError, match="memory_id"):
        method(memory_id)

    assert memory_store.sessions_opened == 0


def test_service_calls_use_repository_as_shared_source_of_truth(
    memory_store: FakeStore,
) -> None:
    first_service = MemoryService()
    created = first_service.remember("Shared durable value")

    second_service = MemoryService()
    recalled = second_service.recall()
    retrieved = second_service.get(created.id)

    assert recalled == [created]
    assert retrieved == created
    assert memory_store.sessions_opened == 3


class EmptyScalarResult:
    def all(self) -> list[object]:
        return []


class CapturingSession:
    def __init__(self) -> None:
        self.statement: Any = None

    def scalars(self, statement: Any) -> EmptyScalarResult:
        self.statement = statement
        return EmptyScalarResult()


def test_repository_builds_filtered_ordered_limited_database_query() -> None:
    session = CapturingSession()
    repository = MemoryRepository(session)  # type: ignore[arg-type]

    assert repository.list_active_memories(
        limit=7,
        memory_type="preference",
    ) == []

    compiled = str(
        session.statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).replace("\n", " ")
    normalized = " ".join(compiled.split())

    assert "WHERE memories.is_active IS true" in normalized
    assert "memories.memory_type = 'preference'" in normalized
    assert (
        "ORDER BY memories.importance DESC, memories.updated_at DESC, "
        "memories.id DESC"
    ) in normalized
    assert normalized.endswith("LIMIT 7")
