from dataclasses import dataclass
from datetime import UTC, datetime

from friday.database.connection import SessionLocal
from friday.database.models import Reminder
from friday.database.repositories import ReminderRepository
from friday.services.task_service import _validate_limit, _validate_positive_id

_TITLE_MAX_LENGTH = 300


@dataclass(frozen=True, slots=True)
class ReminderSnapshot:
    id: int
    title: str
    remind_at: datetime
    recurrence: str | None
    task_id: int | None
    is_enabled: bool
    fired_at: datetime | None
    created_at: datetime


class ReminderService:
    def create(
        self,
        title: str,
        *,
        remind_at: datetime,
        task_id: int | None = None,
        recurrence: str | None = None,
    ) -> ReminderSnapshot:
        title = _normalize_title(title)
        _validate_aware_datetime(remind_at, field_name="remind_at")
        if task_id is not None:
            _validate_positive_id(task_id, field_name="task_id")
        if recurrence is not None:
            raise ValueError("recurrence is not supported in Phase 3")

        with SessionLocal() as session:
            repository = ReminderRepository(session)
            if task_id is not None and not repository.task_exists(task_id):
                raise ValueError(f"Task {task_id} does not exist.")
            reminder = repository.create_reminder(
                title=title,
                remind_at=remind_at,
                recurrence=None,
                task_id=task_id,
            )
            session.commit()
            return _reminder_to_snapshot(reminder)

    def get(self, reminder_id: int) -> ReminderSnapshot | None:
        _validate_positive_id(reminder_id, field_name="reminder_id")
        with SessionLocal() as session:
            reminder = ReminderRepository(session).get_reminder(reminder_id)
            return None if reminder is None else _reminder_to_snapshot(reminder)

    def list_upcoming(self, *, limit: int = 100) -> list[ReminderSnapshot]:
        _validate_limit(limit)
        with SessionLocal() as session:
            reminders = ReminderRepository(session).list_upcoming(limit=limit)
            return [_reminder_to_snapshot(reminder) for reminder in reminders]

    def enable(self, reminder_id: int) -> bool:
        return self._set_enabled(reminder_id, enabled=True)

    def disable(self, reminder_id: int) -> bool:
        return self._set_enabled(reminder_id, enabled=False)

    @staticmethod
    def _set_enabled(reminder_id: int, *, enabled: bool) -> bool:
        _validate_positive_id(reminder_id, field_name="reminder_id")
        with SessionLocal() as session:
            reminder = ReminderRepository(session).set_enabled(
                reminder_id,
                enabled=enabled,
            )
            if reminder is None:
                return False
            session.commit()
            return True

    def claim_due(
        self,
        *,
        now: datetime | None = None,
        limit: int = 20,
    ) -> list[ReminderSnapshot]:
        claim_time = datetime.now(UTC) if now is None else now
        _validate_aware_datetime(claim_time, field_name="now")
        _validate_limit(limit)
        with SessionLocal() as session:
            reminders = ReminderRepository(session).claim_due(
                now=claim_time,
                limit=limit,
            )
            session.commit()
            return [_reminder_to_snapshot(reminder) for reminder in reminders]


def _reminder_to_snapshot(reminder: Reminder) -> ReminderSnapshot:
    return ReminderSnapshot(
        id=reminder.id,
        title=reminder.title,
        remind_at=reminder.remind_at,
        recurrence=reminder.recurrence,
        task_id=reminder.task_id,
        is_enabled=reminder.is_enabled,
        fired_at=reminder.fired_at,
        created_at=reminder.created_at,
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


def _validate_aware_datetime(value: datetime, *, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")  # noqa: TRY004
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
