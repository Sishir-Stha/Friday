from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest

import friday.services.task_service as task_module
from friday.services.task_service import TaskService, TaskStatus

NAIVE_DATETIME = datetime(2026, 1, 1)  # noqa: DTZ001


class FakeSession:
    commits = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def commit(self) -> None:
        type(self).commits += 1


class FakeTaskRepository:
    tasks: ClassVar[dict[int, SimpleNamespace]] = {}
    next_id = 1

    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def create_task(self, **values: Any) -> SimpleNamespace:
        now = datetime.now(UTC)
        task = SimpleNamespace(
            id=self.next_id,
            status="open",
            completed_at=None,
            created_at=now,
            updated_at=now,
            **values,
        )
        type(self).next_id += 1
        self.tasks[task.id] = task
        return task

    def get_task(self, task_id: int) -> SimpleNamespace | None:
        return self.tasks.get(task_id)

    def list_tasks(self, *, status: str | None, limit: int) -> list[SimpleNamespace]:
        tasks = list(self.tasks.values())
        if status is not None:
            tasks = [task for task in tasks if task.status == status]
        return tasks[:limit]

    def update_task(self, task_id: int, **values: Any) -> SimpleNamespace | None:
        task = self.get_task(task_id)
        if task is None:
            return None
        for name, value in values.items():
            setattr(task, name, value)
        task.updated_at = datetime.now(UTC)
        return task

    def set_task_status(
        self,
        task_id: int,
        *,
        status: str,
        completed_at: datetime | None,
    ) -> SimpleNamespace | None:
        task = self.get_task(task_id)
        if task is None:
            return None
        task.status = status
        task.completed_at = completed_at
        task.updated_at = datetime.now(UTC)
        return task


@pytest.fixture(autouse=True)
def fake_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeTaskRepository.tasks = {}
    FakeTaskRepository.next_id = 1
    FakeSession.commits = 0
    monkeypatch.setattr(task_module, "SessionLocal", FakeSession)
    monkeypatch.setattr(task_module, "TaskRepository", FakeTaskRepository)


def test_create_normalizes_and_returns_immutable_snapshot() -> None:
    snapshot = TaskService().create("  Ship Friday  ", description="  Ready  ")

    assert snapshot.title == "Ship Friday"
    assert snapshot.description == "Ready"
    assert snapshot.status is TaskStatus.OPEN
    assert snapshot.priority == 3
    assert FakeSession.commits == 1
    with pytest.raises(FrozenInstanceError):
        snapshot.title = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("title", ["", "   ", None, 1, "x" * 301])
def test_create_rejects_invalid_title(title: Any) -> None:
    with pytest.raises(ValueError, match="title"):
        TaskService().create(title)


@pytest.mark.parametrize("priority", [0, 6, True, 1.5, "3"])
def test_create_rejects_invalid_priority(priority: Any) -> None:
    with pytest.raises(ValueError, match="priority"):
        TaskService().create("Task", priority=priority)


def test_description_empty_becomes_none_and_is_bounded() -> None:
    assert TaskService().create("Task", description="   ").description is None
    with pytest.raises(ValueError, match="description"):
        TaskService().create("Task", description="x" * 4001)


@pytest.mark.parametrize("due_at", [NAIVE_DATETIME, "tomorrow", 1])
def test_create_rejects_invalid_due_datetime(due_at: Any) -> None:
    with pytest.raises(ValueError, match="due_at"):
        TaskService().create("Task", due_at=due_at)


def test_get_and_list_across_independent_service_calls() -> None:
    created = TaskService().create("Task")

    fetched = TaskService().get(created.id)
    listed = TaskService().list(status=TaskStatus.OPEN, limit=10)

    assert fetched == created
    assert listed == [created]


def test_update_changes_only_requested_fields() -> None:
    due = datetime.now(UTC) + timedelta(days=1)
    created = TaskService().create("Original", description="Keep", priority=2)

    updated = TaskService().update(created.id, title=" Updated ", due_at=due)

    assert updated is not None
    assert updated.title == "Updated"
    assert updated.description == "Keep"
    assert updated.priority == 2
    assert updated.due_at == due


def test_start_complete_and_cancel_lifecycle() -> None:
    service = TaskService()
    started = service.create("Started")
    completed = service.create("Completed")
    cancelled = service.create("Cancelled")

    assert service.start(started.id) is True
    assert service.complete(completed.id) is True
    assert service.cancel(cancelled.id) is True

    assert service.get(started.id).status is TaskStatus.IN_PROGRESS  # type: ignore[union-attr]
    completed_snapshot = service.get(completed.id)
    assert completed_snapshot is not None
    assert completed_snapshot.status is TaskStatus.DONE
    assert completed_snapshot.completed_at is not None
    assert completed_snapshot.completed_at.tzinfo is not None
    assert service.get(cancelled.id).status is TaskStatus.CANCELLED  # type: ignore[union-attr]


def test_done_task_cannot_be_reopened_or_cancelled() -> None:
    service = TaskService()
    task = service.create("Done")
    service.complete(task.id)

    assert service.start(task.id) is False
    assert service.cancel(task.id) is False
    assert service.get(task.id).status is TaskStatus.DONE  # type: ignore[union-attr]


def test_missing_ids_return_none_or_false_without_commit() -> None:
    service = TaskService()

    assert service.get(99) is None
    assert service.update(99, title="Missing") is None
    assert service.start(99) is False
    assert service.complete(99) is False
    assert service.cancel(99) is False
    assert FakeSession.commits == 0


@pytest.mark.parametrize("task_id", [0, -1, True, "1"])
def test_invalid_task_ids_are_rejected(task_id: Any) -> None:
    with pytest.raises(ValueError, match="task_id"):
        TaskService().get(task_id)


@pytest.mark.parametrize("limit", [0, 201, True, "10"])
def test_list_rejects_invalid_limit(limit: Any) -> None:
    with pytest.raises(ValueError, match="limit"):
        TaskService().list(limit=limit)


def test_list_rejects_non_enum_status() -> None:
    with pytest.raises(ValueError, match="TaskStatus"):
        TaskService().list(status="open")  # type: ignore[arg-type]
