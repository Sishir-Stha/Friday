from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle, QSystemTrayIcon, QWidget

from friday.services.reminder_service import ReminderService, ReminderSnapshot

logger = logging.getLogger(__name__)

REMINDER_POLL_INTERVAL_MS = 30_000


class _ReminderClaimWorker(QObject):
    completed = Signal(object)
    failed = Signal()

    def __init__(self, reminder_service: ReminderService) -> None:
        super().__init__()
        self._reminder_service = reminder_service

    @Slot()
    def run(self) -> None:
        try:
            reminders = self._reminder_service.claim_due()
        except Exception:
            logger.exception("Reminder polling failed")
            self.failed.emit()
        else:
            self.completed.emit(reminders)


class ReminderNotifier(QObject):
    fallback_notification = Signal(str)

    def __init__(
        self,
        reminder_service: ReminderService,
        parent: QWidget,
        *,
        auto_start: bool = True,
    ) -> None:
        super().__init__(parent)
        self._reminder_service = reminder_service
        self._in_progress = False
        self._thread: QThread | None = None
        self._worker: _ReminderClaimWorker | None = None
        self._tray: QSystemTrayIcon | None = None

        if QSystemTrayIcon.isSystemTrayAvailable():
            app = QApplication.instance()
            icon = (
                app.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
                if app is not None
                else QIcon()
            )
            self._tray = QSystemTrayIcon(icon, parent)
            self._tray.setToolTip("Friday reminders")
            self._tray.show()

        self.timer = QTimer(self)
        self.timer.setInterval(REMINDER_POLL_INTERVAL_MS)
        self.timer.timeout.connect(self.poll_now)
        if auto_start:
            self.timer.start()

    @Slot()
    def poll_now(self) -> None:
        if self._in_progress:
            return
        self._in_progress = True
        thread = QThread(self)
        worker = _ReminderClaimWorker(self._reminder_service)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._notify)
        worker.failed.connect(self._failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _notify(self, result: object) -> None:
        reminders = result if isinstance(result, list) else []
        for reminder in reminders:
            if not isinstance(reminder, ReminderSnapshot):
                continue
            if self._tray is not None:
                self._tray.showMessage("Friday reminder", reminder.title)
            else:
                self.fallback_notification.emit(reminder.title)

    @Slot()
    def _failed(self) -> None:
        pass

    @Slot()
    def _finished(self) -> None:
        self._thread = None
        self._worker = None
        self._in_progress = False

    def stop(self) -> None:
        self.timer.stop()
        if self._tray is not None:
            self._tray.hide()
