from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from time import monotonic, sleep

from friday.services.reminder_service import ReminderSnapshot
from friday.services.task_service import TaskSnapshot, TaskStatus
from friday.ui.secondary_window import OrganizerWindow

NOW = datetime.now(UTC)


def _task(task_id: int = 1) -> TaskSnapshot:
    return TaskSnapshot(
        id=task_id,
        title="Test task",
        description=None,
        status=TaskStatus.OPEN,
        priority=3,
        due_at=None,
        completed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _reminder(reminder_id: int = 2) -> ReminderSnapshot:
    return ReminderSnapshot(
        id=reminder_id,
        title="Test reminder",
        remind_at=NOW + timedelta(hours=1),
        recurrence=None,
        task_id=None,
        is_enabled=True,
        fired_at=None,
        created_at=NOW,
    )


class FakeTasks:
    def __init__(self) -> None:
        self.tasks = [_task()]
        self.actions: list[tuple[str, int]] = []
        self.fail_create = False

    def list(self) -> list[TaskSnapshot]:
        return list(self.tasks)

    def create(self, title: str, **kwargs: object) -> TaskSnapshot:
        if self.fail_create:
            raise ValueError("private task failure")
        created = replace(_task(len(self.tasks) + 1), title=title.strip())
        self.tasks.append(created)
        return created

    def start(self, task_id: int) -> bool:
        return self._change("start", task_id, TaskStatus.IN_PROGRESS)

    def complete(self, task_id: int) -> bool:
        return self._change("complete", task_id, TaskStatus.DONE)

    def cancel(self, task_id: int) -> bool:
        return self._change("cancel", task_id, TaskStatus.CANCELLED)

    def _change(self, action: str, task_id: int, status: TaskStatus) -> bool:
        self.actions.append((action, task_id))
        self.tasks = [
            replace(task, status=status) if task.id == task_id else task
            for task in self.tasks
        ]
        return True


class FakeReminders:
    def __init__(self) -> None:
        self.reminders = [_reminder()]
        self.actions: list[tuple[str, int]] = []

    def list_upcoming(self) -> list[ReminderSnapshot]:
        return list(self.reminders)

    def create(self, title: str, **kwargs: object) -> ReminderSnapshot:
        created = replace(
            _reminder(len(self.reminders) + 2),
            title=title.strip(),
            remind_at=kwargs["remind_at"],
            task_id=kwargs["task_id"],
        )
        self.reminders.append(created)
        return created

    def enable(self, reminder_id: int) -> bool:
        return self._change("enable", reminder_id, True)

    def disable(self, reminder_id: int) -> bool:
        return self._change("disable", reminder_id, False)

    def _change(self, action: str, reminder_id: int, enabled: bool) -> bool:
        self.actions.append((action, reminder_id))
        self.reminders = [
            replace(reminder, is_enabled=enabled)
            if reminder.id == reminder_id
            else reminder
            for reminder in self.reminders
        ]
        return True


def _wait(qapp: object, condition: Callable[[], bool]) -> None:
    deadline = monotonic() + 3
    while not condition() and monotonic() < deadline:
        qapp.processEvents()  # type: ignore[attr-defined]
        sleep(0.005)
    assert condition()


def test_tabs_create_and_task_lifecycle_actions(qapp: object) -> None:
    tasks = FakeTasks()
    reminders = FakeReminders()
    window = OrganizerWindow(tasks, reminders, auto_refresh=False)
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "Tasks",
        "Reminders",
    ]

    window.refresh_tasks()
    _wait(qapp, lambda: window._thread is None)
    assert window.task_list.count() == 1

    window.task_title.setText("Added task")
    window.add_task()
    _wait(qapp, lambda: window._thread is None)
    assert [task.title for task in tasks.tasks] == ["Test task", "Added task"]
    assert window.task_list.count() == 2

    window.task_list.setCurrentRow(0)
    window.complete_task_button.click()
    _wait(qapp, lambda: window._thread is None)
    window.task_list.setCurrentRow(1)
    window.cancel_task_button.click()
    _wait(qapp, lambda: window._thread is None)
    assert tasks.actions == [("complete", 1), ("cancel", 2)]
    window.close()


def test_reminder_create_enable_and_disable(qapp: object) -> None:
    tasks = FakeTasks()
    reminders = FakeReminders()
    window = OrganizerWindow(tasks, reminders, auto_refresh=False)
    window.refresh_reminders()
    _wait(qapp, lambda: window._thread is None)

    window.reminder_title.setText("Added reminder")
    window.add_reminder()
    _wait(qapp, lambda: window._thread is None)
    assert len(reminders.reminders) == 2
    assert reminders.reminders[-1].remind_at.utcoffset() is not None

    window.reminder_list.setCurrentRow(0)
    window.disable_reminder_button.click()
    _wait(qapp, lambda: window._thread is None)
    window.reminder_list.setCurrentRow(0)
    window.enable_reminder_button.click()
    _wait(qapp, lambda: window._thread is None)
    assert reminders.actions == [("disable", 2), ("enable", 2)]
    window.close()


def test_service_failure_is_shown_safely(qapp: object) -> None:
    tasks = FakeTasks()
    tasks.fail_create = True
    window = OrganizerWindow(tasks, FakeReminders(), auto_refresh=False)
    window.task_title.setText("will fail")

    window.add_task()
    _wait(qapp, lambda: window._thread is None)

    assert window.error_label.text() == (
        "The organizer could not complete that action."
    )
    assert "private" not in window.error_label.text()
    window.close()
