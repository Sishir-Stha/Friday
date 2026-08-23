from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest

import friday.services.reminder_service as reminder_module
from friday.services.reminder_service import ReminderService

NAIVE_DATETIME = datetime(2026, 1, 1)  # noqa: DTZ001


class FakeSession:
    commits = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def commit(self) -> None:
        type(self).commits += 1


class FakeReminderRepository:
    reminders: ClassVar[dict[int, SimpleNamespace]] = {}
    tasks: ClassVar[set[int]] = {1}
    next_id = 1

    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def task_exists(self, task_id: int) -> bool:
        return task_id in self.tasks

    def create_reminder(self, **values: Any) -> SimpleNamespace:
        reminder = SimpleNamespace(
            id=self.next_id,
            is_enabled=True,
            fired_at=None,
            created_at=datetime.now(UTC),
            **values,
        )
        type(self).next_id += 1
        self.reminders[reminder.id] = reminder
        return reminder

    def get_reminder(self, reminder_id: int) -> SimpleNamespace | None:
        return self.reminders.get(reminder_id)

    def list_upcoming(self, *, limit: int) -> list[SimpleNamespace]:
        return [
            reminder
            for reminder in self.reminders.values()
            if reminder.is_enabled and reminder.fired_at is None
        ][:limit]

    def set_enabled(
        self,
        reminder_id: int,
        *,
        enabled: bool,
    ) -> SimpleNamespace | None:
        reminder = self.get_reminder(reminder_id)
        if reminder is None:
            return None
        reminder.is_enabled = enabled
        return reminder

    def claim_due(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> list[SimpleNamespace]:
        due = [
            reminder
            for reminder in self.reminders.values()
            if reminder.is_enabled
            and reminder.fired_at is None
            and reminder.remind_at <= now
        ][:limit]
        for reminder in due:
            reminder.is_enabled = False
            reminder.fired_at = now
        return due


@pytest.fixture(autouse=True)
def fake_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeReminderRepository.reminders = {}
    FakeReminderRepository.tasks = {1}
    FakeReminderRepository.next_id = 1
    FakeSession.commits = 0
    monkeypatch.setattr(reminder_module, "SessionLocal", FakeSession)
    monkeypatch.setattr(
        reminder_module,
        "ReminderRepository",
        FakeReminderRepository,
    )


def test_create_normalizes_and_returns_immutable_snapshot() -> None:
    remind_at = datetime.now(UTC) + timedelta(hours=1)

    snapshot = ReminderService().create(
        "  Stand up  ",
        remind_at=remind_at,
        task_id=1,
    )

    assert snapshot.title == "Stand up"
    assert snapshot.remind_at == remind_at
    assert snapshot.task_id == 1
    assert snapshot.recurrence is None
    assert snapshot.is_enabled is True
    with pytest.raises(FrozenInstanceError):
        snapshot.title = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("title", ["", "   ", None, 1, "x" * 301])
def test_create_rejects_invalid_title(title: Any) -> None:
    with pytest.raises(ValueError, match="title"):
        ReminderService().create(title, remind_at=datetime.now(UTC))


@pytest.mark.parametrize("remind_at", [NAIVE_DATETIME, "later", None])
def test_create_requires_timezone_aware_datetime(remind_at: Any) -> None:
    with pytest.raises(ValueError, match="remind_at"):
        ReminderService().create("Reminder", remind_at=remind_at)


def test_create_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ReminderService().create(
            "Reminder",
            remind_at=datetime.now(UTC),
            task_id=999,
        )


def test_recurrence_is_explicitly_rejected() -> None:
    with pytest.raises(ValueError, match="not supported"):
        ReminderService().create(
            "Reminder",
            remind_at=datetime.now(UTC),
            recurrence="daily",
        )


def test_get_and_list_upcoming_across_service_calls() -> None:
    created = ReminderService().create(
        "Reminder",
        remind_at=datetime.now(UTC) + timedelta(hours=1),
    )

    assert ReminderService().get(created.id) == created
    assert ReminderService().list_upcoming() == [created]


def test_enable_disable_and_missing_ids() -> None:
    service = ReminderService()
    reminder = service.create("Reminder", remind_at=datetime.now(UTC))

    assert service.disable(reminder.id) is True
    assert service.get(reminder.id).is_enabled is False  # type: ignore[union-attr]
    assert service.enable(reminder.id) is True
    assert service.get(reminder.id).is_enabled is True  # type: ignore[union-attr]
    assert service.enable(999) is False
    assert service.disable(999) is False


def test_claim_due_marks_fired_disables_and_returns_only_once() -> None:
    now = datetime.now(UTC)
    service = ReminderService()
    reminder = service.create("Due", remind_at=now - timedelta(seconds=1))

    claimed = service.claim_due(now=now)
    claimed_again = service.claim_due(now=now)

    assert len(claimed) == 1
    assert claimed[0].id == reminder.id
    assert claimed[0].is_enabled is False
    assert claimed[0].fired_at == now
    assert claimed_again == []


def test_claim_due_ignores_disabled_future_and_already_fired() -> None:
    now = datetime.now(UTC)
    service = ReminderService()
    disabled = service.create("Disabled", remind_at=now - timedelta(seconds=1))
    future = service.create("Future", remind_at=now + timedelta(hours=1))
    fired = service.create("Fired", remind_at=now - timedelta(seconds=1))
    service.disable(disabled.id)
    FakeReminderRepository.reminders[fired.id].fired_at = now

    assert service.claim_due(now=now) == []
    assert service.get(future.id).is_enabled is True  # type: ignore[union-attr]


@pytest.mark.parametrize("reminder_id", [0, -1, True, "1"])
def test_invalid_reminder_ids_are_rejected(reminder_id: Any) -> None:
    with pytest.raises(ValueError, match="reminder_id"):
        ReminderService().get(reminder_id)


@pytest.mark.parametrize("limit", [0, 201, True, "10"])
def test_invalid_limits_are_rejected(limit: Any) -> None:
    with pytest.raises(ValueError, match="limit"):
        ReminderService().list_upcoming(limit=limit)
