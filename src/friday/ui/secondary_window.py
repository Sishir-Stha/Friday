from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from PySide6.QtCore import QDateTime, QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDateTimeEdit,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from friday.services.reminder_service import ReminderService, ReminderSnapshot
from friday.services.task_service import TaskService, TaskSnapshot

logger = logging.getLogger(__name__)


class _ServiceWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, action: Callable[[], object]) -> None:
        super().__init__()
        self._action = action

    @Slot()
    def run(self) -> None:
        try:
            result = self._action()
        except Exception:
            logger.exception("Organizer service operation failed")
            self.failed.emit("The organizer could not complete that action.")
        else:
            self.succeeded.emit(result)


class OrganizerWindow(QDialog):
    def __init__(
        self,
        task_service: TaskService,
        reminder_service: ReminderService,
        parent: QWidget | None = None,
        *,
        auto_refresh: bool = True,
    ) -> None:
        super().__init__(parent)
        self.task_service = task_service
        self.reminder_service = reminder_service
        self._thread: QThread | None = None
        self._worker: _ServiceWorker | None = None
        self._success_handler: Callable[[object], None] | None = None
        self._refresh_after: str | None = None

        self.setWindowTitle("Tasks & Reminders")
        self.resize(720, 560)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("organizerTabs")
        self.tabs.addTab(self._build_tasks_tab(), "Tasks")
        self.tabs.addTab(self._build_reminders_tab(), "Reminders")
        self.error_label = QLabel()
        self.error_label.setObjectName("organizerNotice")
        self.error_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.error_label)

        if auto_refresh:
            self.refresh_tasks()

    def _build_tasks_tab(self) -> QWidget:
        tab = QWidget()
        self.task_list = QListWidget()
        self.task_list.setObjectName("taskList")
        self.task_title = QLineEdit()
        self.task_title.setObjectName("taskTitle")
        self.task_description = QPlainTextEdit()
        self.task_description.setMaximumHeight(70)
        self.task_priority = QSpinBox()
        self.task_priority.setRange(1, 5)
        self.task_priority.setValue(3)
        self.task_due_enabled = QCheckBox("Set due date")
        self.task_due = QDateTimeEdit(QDateTime.currentDateTime().addSecs(3600))
        self.task_due.setCalendarPopup(True)
        self.task_due.setEnabled(False)
        self.task_due_enabled.toggled.connect(self.task_due.setEnabled)

        form = QFormLayout()
        form.addRow("Title", self.task_title)
        form.addRow("Description", self.task_description)
        form.addRow("Priority", self.task_priority)
        form.addRow(self.task_due_enabled, self.task_due)

        self.add_task_button = QPushButton("Add Task")
        self.start_task_button = QPushButton("Start")
        self.complete_task_button = QPushButton("Complete")
        self.cancel_task_button = QPushButton("Cancel")
        self.refresh_tasks_button = QPushButton("Refresh")
        buttons = QHBoxLayout()
        for button in (
            self.add_task_button,
            self.start_task_button,
            self.complete_task_button,
            self.cancel_task_button,
            self.refresh_tasks_button,
        ):
            buttons.addWidget(button)

        layout = QVBoxLayout(tab)
        layout.addWidget(self.task_list, 1)
        layout.addLayout(form)
        layout.addLayout(buttons)

        self.add_task_button.clicked.connect(self.add_task)
        self.start_task_button.clicked.connect(
            lambda: self._change_selected_task("start")
        )
        self.complete_task_button.clicked.connect(
            lambda: self._change_selected_task("complete")
        )
        self.cancel_task_button.clicked.connect(
            lambda: self._change_selected_task("cancel")
        )
        self.refresh_tasks_button.clicked.connect(self.refresh_tasks)
        return tab

    def _build_reminders_tab(self) -> QWidget:
        tab = QWidget()
        self.reminder_list = QListWidget()
        self.reminder_list.setObjectName("reminderList")
        self.reminder_title = QLineEdit()
        self.reminder_title.setObjectName("reminderTitle")
        self.reminder_at = QDateTimeEdit(QDateTime.currentDateTime().addSecs(3600))
        self.reminder_at.setCalendarPopup(True)
        self.reminder_task_id = QSpinBox()
        self.reminder_task_id.setRange(0, 2_147_483_647)
        self.reminder_task_id.setSpecialValueText("None")

        form = QFormLayout()
        form.addRow("Title", self.reminder_title)
        form.addRow("Remind at", self.reminder_at)
        form.addRow("Linked task", self.reminder_task_id)

        self.add_reminder_button = QPushButton("Add Reminder")
        self.enable_reminder_button = QPushButton("Enable")
        self.disable_reminder_button = QPushButton("Disable")
        self.refresh_reminders_button = QPushButton("Refresh")
        buttons = QHBoxLayout()
        for button in (
            self.add_reminder_button,
            self.enable_reminder_button,
            self.disable_reminder_button,
            self.refresh_reminders_button,
        ):
            buttons.addWidget(button)

        layout = QVBoxLayout(tab)
        layout.addWidget(self.reminder_list, 1)
        layout.addLayout(form)
        layout.addLayout(buttons)

        self.add_reminder_button.clicked.connect(self.add_reminder)
        self.enable_reminder_button.clicked.connect(
            lambda: self._change_selected_reminder("enable")
        )
        self.disable_reminder_button.clicked.connect(
            lambda: self._change_selected_reminder("disable")
        )
        self.refresh_reminders_button.clicked.connect(self.refresh_reminders)
        return tab

    @Slot()
    def refresh_tasks(self) -> None:
        self._run(
            lambda: self.task_service.list(),
            lambda result: self._render_tasks(result),
        )

    @Slot()
    def refresh_reminders(self) -> None:
        self._run(
            lambda: self.reminder_service.list_upcoming(),
            lambda result: self._render_reminders(result),
        )

    @Slot()
    def add_task(self) -> None:
        due_at = (
            _aware_local_datetime(self.task_due.dateTime())
            if self.task_due_enabled.isChecked()
            else None
        )
        self._refresh_after = "tasks"
        self._run(
            lambda: self.task_service.create(
                self.task_title.text(),
                description=self.task_description.toPlainText(),
                priority=self.task_priority.value(),
                due_at=due_at,
            ),
            self._task_created,
        )

    def _task_created(self, result: object) -> None:
        self.task_title.clear()
        self.task_description.clear()

    def _change_selected_task(self, action: str) -> None:
        task_id = _selected_id(self.task_list)
        if task_id is None:
            self.error_label.setText("Select a task first.")
            return
        operation = getattr(self.task_service, action)
        self._refresh_after = "tasks"
        self._run(lambda: operation(task_id), lambda result: None)

    @Slot()
    def add_reminder(self) -> None:
        task_id = self.reminder_task_id.value() or None
        self._refresh_after = "reminders"
        self._run(
            lambda: self.reminder_service.create(
                self.reminder_title.text(),
                remind_at=_aware_local_datetime(self.reminder_at.dateTime()),
                task_id=task_id,
            ),
            self._reminder_created,
        )

    def _reminder_created(self, result: object) -> None:
        self.reminder_title.clear()

    def _change_selected_reminder(self, action: str) -> None:
        reminder_id = _selected_id(self.reminder_list)
        if reminder_id is None:
            self.error_label.setText("Select a reminder first.")
            return
        operation = getattr(self.reminder_service, action)
        self._refresh_after = "reminders"
        self._run(lambda: operation(reminder_id), lambda result: None)

    def _run(
        self,
        action: Callable[[], object],
        on_success: Callable[[object], None],
    ) -> None:
        if self._thread is not None:
            return
        self.error_label.clear()
        self._success_handler = on_success
        self._set_actions_enabled(False)
        thread = QThread(self)
        worker = _ServiceWorker(action)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._operation_succeeded)
        worker.failed.connect(self._operation_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._operation_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _operation_succeeded(self, result: object) -> None:
        if self._success_handler is not None:
            self._success_handler(result)

    @Slot(str)
    def _operation_failed(self, message: str) -> None:
        self.error_label.setText(message)
        self._refresh_after = None

    @Slot()
    def _operation_finished(self) -> None:
        refresh_after = self._refresh_after
        self._thread = None
        self._worker = None
        self._success_handler = None
        self._refresh_after = None
        self._set_actions_enabled(True)
        if refresh_after == "tasks":
            self.refresh_tasks()
        elif refresh_after == "reminders":
            self.refresh_reminders()

    def _render_tasks(self, result: object) -> None:
        tasks = result if isinstance(result, list) else []
        self.task_list.clear()
        for task in tasks:
            if not isinstance(task, TaskSnapshot):
                continue
            due = _format_local_datetime(task.due_at)
            suffix = f" · due {due}" if due else ""
            item = QListWidgetItem(
                f"[{task.status.value}] P{task.priority} · {task.title}{suffix}"
            )
            item.setData(Qt.ItemDataRole.UserRole, task.id)
            self.task_list.addItem(item)

    def _render_reminders(self, result: object) -> None:
        reminders = result if isinstance(result, list) else []
        self.reminder_list.clear()
        for reminder in reminders:
            if not isinstance(reminder, ReminderSnapshot):
                continue
            enabled = "enabled" if reminder.is_enabled else "disabled"
            item = QListWidgetItem(
                f"[{enabled}] {reminder.title} · "
                f"{_format_local_datetime(reminder.remind_at)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, reminder.id)
            self.reminder_list.addItem(item)

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.add_task_button,
            self.start_task_button,
            self.complete_task_button,
            self.cancel_task_button,
            self.refresh_tasks_button,
            self.add_reminder_button,
            self.enable_reminder_button,
            self.disable_reminder_button,
            self.refresh_reminders_button,
        ):
            button.setEnabled(enabled)


def _selected_id(widget: QListWidget) -> int | None:
    item = widget.currentItem()
    if item is None:
        return None
    value = item.data(Qt.ItemDataRole.UserRole)
    return value if isinstance(value, int) else None


def _aware_local_datetime(value: QDateTime) -> datetime:
    result = value.toPython()
    if result.tzinfo is None or result.utcoffset() is None:
        return result.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return result.astimezone()


def _format_local_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone().strftime("%Y-%m-%d %H:%M")
