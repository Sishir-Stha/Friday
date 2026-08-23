from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from friday.database.connection import SessionLocal
from friday.database.models import Task
from friday.services.task_service import TaskService, TaskStatus

pytestmark = pytest.mark.integration


def test_task_service_postgresql_lifecycle() -> None:
    service = TaskService()
    title = f"Friday task integration {uuid4()}"
    task_id: int | None = None

    try:
        created = service.create(
            title,
            description="temporary integration row",
            priority=4,
            due_at=datetime.now(UTC) + timedelta(days=1),
        )
        task_id = created.id
        assert created.status is TaskStatus.OPEN

        retrieved = service.get(task_id)
        assert retrieved is not None and retrieved.title == title
        assert task_id in {task.id for task in service.list(limit=200)}

        updated = service.update(
            task_id,
            title=f"{title} updated",
            description=None,
            priority=5,
        )
        assert updated is not None
        assert updated.title.endswith(" updated")
        assert updated.description is None
        assert updated.priority == 5

        assert service.start(task_id) is True
        started = service.get(task_id)
        assert started is not None
        assert started.status is TaskStatus.IN_PROGRESS

        assert service.complete(task_id) is True
        completed = service.get(task_id)
        assert completed is not None
        assert completed.status is TaskStatus.DONE
        assert completed.completed_at is not None
    finally:
        if task_id is not None:
            with SessionLocal() as session:
                task = session.get(Task, task_id)
                if task is not None:
                    session.delete(task)
                    session.commit()
            with SessionLocal() as session:
                assert session.get(Task, task_id) is None
