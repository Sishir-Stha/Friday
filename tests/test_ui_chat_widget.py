from __future__ import annotations

from collections.abc import Callable
from threading import Event
from time import monotonic, sleep

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from friday.llm.ollama_client import OllamaResponse
from friday.ui.chat_widget import ChatWidget


class FakeConversationService:
    def __init__(self, *, gate: Event | None = None, error: bool = False) -> None:
        self.gate = gate
        self.error = error
        self.created = 0
        self.messages: list[tuple[int, str]] = []

    def create_conversation(self, title: str | None = None) -> int:
        self.created += 1
        return 41

    def send_message(self, conversation_id: int, content: str) -> OllamaResponse:
        self.messages.append((conversation_id, content))
        if self.gate is not None:
            self.gate.wait(timeout=2)
        if self.error:
            raise RuntimeError("private diagnostic")
        return OllamaResponse(content="Hello")


def _wait(qapp: object, condition: Callable[[], bool]) -> None:
    deadline = monotonic() + 3
    while not condition() and monotonic() < deadline:
        qapp.processEvents()  # type: ignore[attr-defined]
        sleep(0.005)
    assert condition()


def test_blank_is_ignored_and_success_is_displayed(qapp: object) -> None:
    service = FakeConversationService()
    widget = ChatWidget(service)
    widget.composer.setPlainText("   ")
    widget.send_current_message()
    assert service.created == 0

    widget.composer.setPlainText("Say hello")
    widget.send_current_message()
    assert "You\nSay hello" in widget.message_display.toPlainText()
    assert not widget.send_button.isEnabled()
    _wait(qapp, lambda: not widget.is_busy)

    assert service.messages == [(41, "Say hello")]
    assert "Friday\nHello" in widget.message_display.toPlainText()
    assert widget.send_button.isEnabled()
    assert widget.composer.isEnabled()
    _wait(qapp, lambda: widget._thread is None)
    widget.close()


def test_overlapping_send_is_prevented(qapp: object) -> None:
    gate = Event()
    service = FakeConversationService(gate=gate)
    widget = ChatWidget(service)
    widget.composer.setPlainText("first")
    widget.send_current_message()
    widget.composer.setPlainText("second")
    widget.send_current_message()

    _wait(qapp, lambda: len(service.messages) == 1)
    gate.set()
    _wait(qapp, lambda: not widget.is_busy)
    assert service.messages == [(41, "first")]
    _wait(qapp, lambda: widget._thread is None)
    widget.close()


def test_error_is_safe_and_composer_recovers(qapp: object) -> None:
    widget = ChatWidget(FakeConversationService(error=True))
    widget.composer.setPlainText("hello")
    widget.send_current_message()
    _wait(qapp, lambda: not widget.is_busy)

    text = widget.message_display.toPlainText()
    assert "Friday encountered an error." in text
    assert "private diagnostic" not in text
    assert widget.composer.isEnabled()
    _wait(qapp, lambda: widget._thread is None)
    widget.close()


def test_enter_sends_and_shift_enter_inserts_newline(qapp: object) -> None:
    service = FakeConversationService()
    widget = ChatWidget(service)
    widget.show()
    widget.composer.setFocus()
    QTest.keyClicks(widget.composer, "hello")
    QTest.keyClick(widget.composer, Qt.Key.Key_Return)
    _wait(qapp, lambda: not widget.is_busy)
    assert service.messages == [(41, "hello")]

    QTest.keyClicks(widget.composer, "line one")
    QTest.keyClick(
        widget.composer,
        Qt.Key.Key_Return,
        Qt.KeyboardModifier.ShiftModifier,
    )
    QTest.keyClicks(widget.composer, "line two")
    assert widget.composer.toPlainText() == "line one\nline two"
    _wait(qapp, lambda: widget._thread is None)
    widget.close()
