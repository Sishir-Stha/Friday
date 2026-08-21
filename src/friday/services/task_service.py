from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from friday.database.connection import SessionLocal
from friday.database.models import Task
from friday.database.repositories import TaskRepository

_TITLE_MAX_LENGTH = 300
_DESCRIPTION_MAX_LENGTH = 4000


class TaskStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    id: int
    title: str
    description: str | None
    status: TaskStatus
    priority: int
    due_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class _UnsetType:
    __slots__ = ()


UNSET = _UnsetType()


class TaskService:
    def create(
        self,
        title: str,
        *,
        description: str | None = None,
        priority: int = 3,
        due_at: datetime | None = None,
    ) -> TaskSnapshot:
        title = _normalize_title(title)
        description = _normalize_description(description)
        _validate_priority(priority)
        _validate_optional_aware_datetime(due_at, field_name="due_at")

        with SessionLocal() as session:
            repository = TaskRepository(session)
            task = repository.create_task(
                title=title,
                description=description,
                priority=priority,
                due_at=due_at,
            )
            session.commit()
            return _task_to_snapshot(task)

    def get(self, task_id: int) -> TaskSnapshot | None:
        _validate_positive_id(task_id, field_name="task_id")
        with SessionLocal() as session:
            task = TaskRepository(session).get_task(task_id)
            return None if task is None else _task_to_snapshot(task)

    def list(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> list[TaskSnapshot]:
        if status is not None and not isinstance(status, TaskStatus):
            raise ValueError("status must be a TaskStatus value")
        _validate_limit(limit)
        with SessionLocal() as session:
            tasks = TaskRepository(session).list_tasks(
                status=None if status is None else status.value,
                limit=limit,
            )
            return [_task_to_snapshot(task) for task in tasks]

    def update(
        self,
        task_id: int,
        *,
        title: str | _UnsetType = UNSET,
        description: str | None | _UnsetType = UNSET,
        priority: int | _UnsetType = UNSET,
        due_at: datetime | None | _UnsetType = UNSET,
    ) -> TaskSnapshot | None:
        _validate_positive_id(task_id, field_name="task_id")
        with SessionLocal() as session:
            repository = TaskRepository(session)
            current = repository.get_task(task_id)
            if current is None:
                return None

            next_title = current.title if isinstance(title, _UnsetType) else _normalize_title(title)
            next_description = (
                current.description
                if isinstance(description, _UnsetType)
                else _normalize_description(description)
            )
            next_priority = current.priority if isinstance(priority, _UnsetType) else priority
            _validate_priority(next_priority)
            next_due_at = current.due_at if isinstance(due_at, _UnsetType) else due_at
            _validate_optional_aware_datetime(next_due_at, field_name="due_at")

            task = repository.update_task(
                task_id,
                title=next_title,
                description=next_description,
                priority=next_priority,
                due_at=next_due_at,
            )
            session.commit()
            return None if task is None else _task_to_snapshot(task)

    def start(self, task_id: int) -> bool:
        return self._set_status(task_id, TaskStatus.IN_PROGRESS)

    def complete(self, task_id: int) -> bool:
        return self._set_status(task_id, TaskStatus.DONE)

    def cancel(self, task_id: int) -> bool:
        return self._set_status(task_id, TaskStatus.CANCELLED)

    @staticmethod
    def _set_status(task_id: int, status: TaskStatus) -> bool:
        _validate_positive_id(task_id, field_name="task_id")
        with SessionLocal() as session:
            repository = TaskRepository(session)
            current = repository.get_task(task_id)
            if current is None:
                return False
            if current.status == TaskStatus.DONE.value and status is not TaskStatus.DONE:
                return False

            completed_at = datetime.now(UTC) if status is TaskStatus.DONE else None
            repository.set_task_status(
                task_id,
                status=status.value,
                completed_at=completed_at,
            )
            session.commit()
            return True


def _task_to_snapshot(task: Task) -> TaskSnapshot:
    return TaskSnapshot(
        id=task.id,
        title=task.title,
        description=task.description,
        status=TaskStatus(task.status),
        priority=task.priority,
        due_at=task.due_at,
        completed_at=task.completed_at,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def _normalize_title(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("title must be a string")  # noqa: TRY004
    normalized = value.strip()
    if not normalized:
        raise ValueError("title must not be empty")
    if len(normalized) > _TITLE_MAX_LENGTH:
        raise ValueError(f"title must be at most {_TITLE_MAX_LENGTH} characters")
    return normalized


def _normalize_description(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("description must be a string or None")  # noqa: TRY004
    normalized = value.strip()
    if len(normalized) > _DESCRIPTION_MAX_LENGTH:
        raise ValueError(
            f"description must be at most {_DESCRIPTION_MAX_LENGTH} characters"
        )
    return normalized or None


def _validate_priority(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError("priority must be an integer between 1 and 5")


def _validate_positive_id(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 200:
        raise ValueError("limit must be an integer between 1 and 200")


def _validate_optional_aware_datetime(
    value: datetime | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime or None")  # noqa: TRY004
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
