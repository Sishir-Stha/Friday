from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Event, Lock

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox, QWidget

from friday.tools.models import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass(slots=True, eq=False)
class _ApprovalRequest:
    tool: ToolDefinition
    arguments: Mapping[str, object]
    resolved: Event = field(default_factory=Event)
    approved: bool = False


class ToolApprovalBridge(QObject):
    """Synchronously bridge a worker request to a GUI-thread decision."""

    approval_requested = Signal(object)

    def __init__(
        self,
        *,
        parent_widget: QWidget | None = None,
        prompt: Callable[[ToolDefinition, Mapping[str, object]], bool]
        | None = None,
    ) -> None:
        super().__init__()
        self._parent_widget = parent_widget
        self._prompt = prompt if prompt is not None else self._show_dialog
        self._lock = Lock()
        self._pending: set[_ApprovalRequest] = set()
        self._closing = False
        self.approval_requested.connect(self._resolve_on_gui_thread)

    def set_parent_widget(self, widget: QWidget) -> None:
        self._parent_widget = widget

    def __call__(
        self,
        tool: ToolDefinition,
        arguments: Mapping[str, object],
    ) -> bool:
        request = _ApprovalRequest(tool=tool, arguments=dict(arguments))
        with self._lock:
            if self._closing:
                return False
            self._pending.add(request)

        if QThread.currentThread() is self.thread():
            self._resolve_on_gui_thread(request)
        else:
            self.approval_requested.emit(request)

        while not request.resolved.wait(0.1):
            with self._lock:
                if self._closing:
                    break

        with self._lock:
            self._pending.discard(request)
        return request.approved if request.resolved.is_set() else False

    @Slot(object)
    def _resolve_on_gui_thread(self, request: _ApprovalRequest) -> None:
        with self._lock:
            closing = self._closing
        if not closing:
            try:
                request.approved = bool(
                    self._prompt(request.tool, request.arguments)
                )
            except Exception:
                logger.exception("Tool approval prompt failed closed")
                request.approved = False
        request.resolved.set()

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            pending = tuple(self._pending)
        for request in pending:
            request.approved = False
            request.resolved.set()

    def _show_dialog(
        self,
        tool: ToolDefinition,
        arguments: Mapping[str, object],
    ) -> bool:
        dialog = QMessageBox(self._parent_widget)
        dialog.setWindowTitle("Friday tool approval")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText(f"Tool: {tool.name}")
        if "app_name" in arguments:
            dialog.setInformativeText(
                f"Requested app: {arguments['app_name']}"
            )
        else:
            dialog.setInformativeText("Allow this requested action once?")
        allow_button = dialog.addButton(
            "Allow once",
            QMessageBox.ButtonRole.AcceptRole,
        )
        dialog.addButton("Deny", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        return dialog.clickedButton() is allow_button
