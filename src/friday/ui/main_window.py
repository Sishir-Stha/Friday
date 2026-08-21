from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from friday.core.runtime import FridayRuntime, build_friday_runtime
from friday.services.assistant_state import AssistantState
from friday.ui.approval_bridge import ToolApprovalBridge
from friday.ui.chat_widget import ChatWidget
from friday.ui.reminder_notifier import ReminderNotifier
from friday.ui.secondary_window import OrganizerWindow
from friday.ui.system_status import SystemStatusWidget

logger = logging.getLogger(__name__)

_STATE_TEXT = {
    AssistantState.IDLE: "Ready",
    AssistantState.PROCESSING: "Processing",
    AssistantState.STREAMING: "Responding",
    AssistantState.TOOL_RUNNING: "Running tool",
    AssistantState.OFFLINE: "Offline",
    AssistantState.ERROR: "Error",
    AssistantState.LISTENING: "Listening",
    AssistantState.SPEAKING: "Speaking",
}


class _HealthWorker(QObject):
    available = Signal()
    unavailable = Signal()

    def __init__(self, health_check: Callable[[], object]) -> None:
        super().__init__()
        self._health_check = health_check

    @Slot()
    def run(self) -> None:
        try:
            self._health_check()
        except Exception:  # noqa: BLE001
            logger.info("Ollama startup health check reported offline")
            self.unavailable.emit()
        else:
            self.available.emit()


class FridayMainWindow(QMainWindow):
    def __init__(
        self,
        runtime: FridayRuntime,
        *,
        approval_bridge: ToolApprovalBridge | None = None,
        start_background_workers: bool = True,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.approval_bridge = approval_bridge
        self.organizer: OrganizerWindow | None = None
        self._health_thread: QThread | None = None
        self._health_worker: _HealthWorker | None = None
        self._health_offline = False

        self.setWindowTitle("Friday")
        self.resize(920, 680)
        self.setMinimumSize(680, 500)

        self.identity_label = QLabel("FRIDAY")
        self.identity_label.setObjectName("fridayIdentity")
        self.state_label = QLabel("Ready")
        self.state_label.setObjectName("assistantState")
        mode = runtime.settings.ai_mode.strip().title()
        self.ai_label = QLabel(f"{mode} • {runtime.settings.ollama_model}")
        self.ai_label.setObjectName("aiMode")
        self.menu_button = QToolButton()
        self.menu_button.setText("⋯")
        self.menu_button.setObjectName("overflowMenu")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        menu = QMenu(self.menu_button)
        self.organizer_action = QAction("Tasks & Reminders", self)
        self.new_chat_action = QAction("New Chat", self)
        menu.addAction(self.organizer_action)
        menu.addAction(self.new_chat_action)
        self.menu_button.setMenu(menu)

        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(self.identity_label)
        header.addWidget(self.state_label)
        header.addStretch(1)
        header.addWidget(self.ai_label)
        header.addWidget(self.menu_button)

        self.chat_widget = ChatWidget(runtime.conversation_service)
        self.chat_widget.setObjectName("chatArea")
        self.system_status = SystemStatusWidget(
            runtime.system_monitor,
            auto_start=start_background_workers,
        )
        self.system_status.setObjectName("systemStatus")
        self.system_status.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addWidget(self.chat_widget, 1)
        layout.addWidget(self.system_status)
        self.setCentralWidget(central)

        self.organizer_action.triggered.connect(self.open_organizer)
        self.new_chat_action.triggered.connect(self.chat_widget.new_chat)

        self.state_timer = QTimer(self)
        self.state_timer.setInterval(250)
        self.state_timer.timeout.connect(self.refresh_state)
        self.state_timer.start()

        self.reminder_notifier = ReminderNotifier(
            runtime.reminder_service,
            self,
            auto_start=start_background_workers,
        )
        self.reminder_notifier.fallback_notification.connect(
            lambda title: self.statusBar().showMessage(
                f"Reminder: {title}",
                10_000,
            )
        )

        self.setStyleSheet(_STYLE_SHEET)
        if start_background_workers:
            QTimer.singleShot(0, self._start_health_check)

    @Slot()
    def refresh_state(self) -> None:
        if self.chat_widget.is_busy:
            self._health_offline = False
        if self._health_offline:
            self.state_label.setText("Offline")
            return
        state = self.runtime.state_machine.get()
        self.state_label.setText(_STATE_TEXT[state])

    @Slot()
    def open_organizer(self) -> None:
        if self.organizer is None:
            self.organizer = OrganizerWindow(
                self.runtime.task_service,
                self.runtime.reminder_service,
                self,
            )
        self.organizer.show()
        self.organizer.raise_()
        self.organizer.activateWindow()

    @Slot()
    def _start_health_check(self) -> None:
        if self._health_thread is not None:
            return
        thread = QThread(self)
        worker = _HealthWorker(
            self.runtime.conversation_service.ollama.health_check
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.available.connect(self._health_available)
        worker.unavailable.connect(self._health_unavailable)
        worker.available.connect(thread.quit)
        worker.unavailable.connect(thread.quit)
        worker.available.connect(worker.deleteLater)
        worker.unavailable.connect(worker.deleteLater)
        thread.finished.connect(self._health_finished)
        thread.finished.connect(thread.deleteLater)
        self._health_thread = thread
        self._health_worker = worker
        thread.start()

    @Slot()
    def _health_available(self) -> None:
        self._health_offline = False
        self.refresh_state()

    @Slot()
    def _health_unavailable(self) -> None:
        self._health_offline = True
        self.refresh_state()

    @Slot()
    def _health_finished(self) -> None:
        self._health_thread = None
        self._health_worker = None

    def closeEvent(self, event: QCloseEvent) -> None:
        self.state_timer.stop()
        self.system_status.stop()
        self.reminder_notifier.stop()
        if self.approval_bridge is not None:
            self.approval_bridge.shutdown()
        super().closeEvent(event)


def run_app() -> None:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)

    approval_bridge = ToolApprovalBridge()
    runtime = build_friday_runtime(tool_approval_handler=approval_bridge)
    window = FridayMainWindow(runtime, approval_bridge=approval_bridge)
    approval_bridge.set_parent_widget(window)
    app.aboutToQuit.connect(approval_bridge.shutdown)
    window.show()

    exit_code = app.exec()
    if owns_app:
        raise SystemExit(exit_code)


_STYLE_SHEET = """
QMainWindow, QDialog, QWidget {
    background: #17191d;
    color: #e8eaed;
    font-size: 14px;
}
#fridayIdentity {
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 2px;
}
#assistantState, #aiMode, #cpuStatus, #ramStatus, #gpuStatus {
    color: #aeb4bd;
}
QPlainTextEdit, QLineEdit, QListWidget, QDateTimeEdit, QSpinBox {
    background: #202329;
    border: 1px solid #30343b;
    border-radius: 7px;
    padding: 8px;
    selection-background-color: #3b6f9f;
}
#conversationDisplay { padding: 15px; }
QPushButton, QToolButton {
    background: #2d5f89;
    border: 0;
    border-radius: 6px;
    padding: 8px 14px;
}
QPushButton:disabled { background: #34383f; color: #81868e; }
QPushButton:hover, QToolButton:hover { background: #386f9e; }
QTabWidget::pane { border: 1px solid #30343b; }
QTabBar::tab { padding: 8px 16px; background: #202329; }
QTabBar::tab:selected { background: #2d5f89; }
"""
