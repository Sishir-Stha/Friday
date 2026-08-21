from __future__ import annotations

import logging
from typing import Protocol

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from friday.llm.ollama_client import OllamaResponse, OllamaUnavailableError

logger = logging.getLogger(__name__)


class ConversationServiceLike(Protocol):
    def create_conversation(self, title: str | None = None) -> int: ...

    def send_message(self, conversation_id: int, content: str) -> OllamaResponse: ...


class MessageComposer(QPlainTextEdit):
    submit_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submit_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class ChatWorker(QObject):
    started = Signal()
    completed = Signal(str, int)
    failed = Signal(str)

    def __init__(
        self,
        service: ConversationServiceLike,
        conversation_id: int | None,
        content: str,
    ) -> None:
        super().__init__()
        self._service = service
        self._conversation_id = conversation_id
        self._content = content

    @Slot()
    def run(self) -> None:
        self.started.emit()
        try:
            conversation_id = self._conversation_id
            if conversation_id is None:
                conversation_id = self._service.create_conversation()
            response = self._service.send_message(conversation_id, self._content)
        except OllamaUnavailableError:
            logger.exception("Desktop chat could not reach Ollama")
            self.failed.emit("Local AI service is offline.")
        except Exception:
            logger.exception("Desktop chat request failed")
            self.failed.emit("Friday encountered an error.")
        else:
            self.completed.emit(response.content, conversation_id)


class ChatWidget(QWidget):
    def __init__(
        self,
        conversation_service: ConversationServiceLike,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.conversation_service = conversation_service
        self.conversation_id: int | None = None
        self._busy = False
        self._thread: QThread | None = None
        self._worker: ChatWorker | None = None

        self.message_display = QPlainTextEdit()
        self.message_display.setObjectName("conversationDisplay")
        self.message_display.setReadOnly(True)
        self.message_display.setPlaceholderText("Ask Friday anything...")

        self.composer = MessageComposer()
        self.composer.setObjectName("messageComposer")
        self.composer.setPlaceholderText("Message Friday")
        self.composer.setMaximumHeight(100)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("sendButton")

        composer_layout = QHBoxLayout()
        composer_layout.addWidget(self.composer, 1)
        composer_layout.addWidget(self.send_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.message_display, 1)
        layout.addLayout(composer_layout)

        self.send_button.clicked.connect(self.send_current_message)
        self.composer.submit_requested.connect(self.send_current_message)

    @property
    def is_busy(self) -> bool:
        return self._busy

    @Slot()
    def send_current_message(self) -> None:
        content = self.composer.toPlainText().strip()
        if self._busy or self._thread is not None or not content:
            return

        self._append_message("You", content)
        self.composer.clear()
        self._set_busy(True)

        thread = QThread(self)
        worker = ChatWorker(
            self.conversation_service,
            self.conversation_id,
            content,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(str, int)
    def _on_completed(self, content: str, conversation_id: int) -> None:
        self.conversation_id = conversation_id
        self._append_message("Friday", content)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._append_message("System", message)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def new_chat(self) -> bool:
        if self._busy:
            return False
        self.conversation_id = None
        self.message_display.clear()
        self.composer.clear()
        return True

    def _append_message(self, speaker: str, content: str) -> None:
        if self.message_display.toPlainText():
            self.message_display.appendPlainText("")
        self.message_display.appendPlainText(f"{speaker}\n{content}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_button.setEnabled(not busy)
        self.composer.setEnabled(not busy)
        if not busy:
            self.composer.setFocus()
