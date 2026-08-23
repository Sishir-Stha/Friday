from __future__ import annotations

from collections.abc import Callable
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

from PySide6.QtWidgets import QLabel

from friday.llm.ollama_client import OllamaResponse
from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.system.monitor import SystemMetricsSnapshot
from friday.ui.main_window import FridayMainWindow


class FakeConversation:
    def __init__(
        self,
        *,
        gate: Event | None = None,
        health_gate: Event | None = None,
    ) -> None:
        self.gate = gate
        self.started = Event()
        self.health_started = Event()

        def health_check() -> object:
            self.health_started.set()
            if health_gate is not None:
                health_gate.wait(timeout=1)
            return object()

        self.ollama = SimpleNamespace(health_check=lambda: object())
        self.ollama.health_check = health_check

    def create_conversation(self, title: str | None = None) -> int:
        return 1

    def send_message(self, conversation_id: int, content: str) -> OllamaResponse:
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=1)
        return OllamaResponse("ok")


class FakeMonitor:
    def __init__(self, gate: Event | None = None) -> None:
        self.gate = gate
        self.started = Event()

    def get_system_metrics(self) -> SystemMetricsSnapshot:
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=1)
        return SystemMetricsSnapshot(0, 0, 0, 0, 0, ())


class FakeTaskService:
    def list(self) -> list[object]:
        return []


class FakeReminderService:
    def list_upcoming(self) -> list[object]:
        return []

    def claim_due(self) -> list[object]:
        return []


def _runtime(
    *,
    conversation: FakeConversation | None = None,
    monitor: FakeMonitor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(ai_mode="local", ollama_model="test-model"),
        state_machine=AssistantStateMachine(),
        conversation_service=conversation or FakeConversation(),
        system_monitor=monitor or FakeMonitor(),
        task_service=FakeTaskService(),
        reminder_service=FakeReminderService(),
    )


def _wait(qapp: object, condition: Callable[[], bool]) -> None:
    deadline = monotonic() + 2
    while not condition() and monotonic() < deadline:
        qapp.processEvents()  # type: ignore[attr-defined]
        sleep(0.005)
    assert condition()


def test_main_window_is_minimal_and_opens_organizer_on_demand(qapp: object) -> None:
    runtime = _runtime()
    window = FridayMainWindow(runtime, start_background_workers=False)
    window.show()
    qapp.processEvents()  # type: ignore[attr-defined]

    assert window.windowTitle() == "Friday"
    assert window.identity_label.text() == "FRIDAY"
    assert window.chat_widget.isVisible()
    assert window.chat_widget.composer.isVisible()
    assert window.system_status.isVisible()
    assert window.ai_label.text() == "Local • test-model"
    assert window.organizer is None
    visible_text = " ".join(
        label.text() for label in window.findChildren(QLabel) if label.isVisible()
    )
    assert "Recent Activity" not in visible_text
    assert "Task list" not in visible_text
    assert "Reminder list" not in visible_text

    window.organizer_action.trigger()
    assert window.organizer is not None and window.organizer.isVisible()
    _wait(qapp, lambda: window.organizer._thread is None)
    window.close()


def test_new_chat_clears_visible_conversation(qapp: object) -> None:
    window = FridayMainWindow(_runtime(), start_background_workers=False)
    window.chat_widget.message_display.setPlainText("old conversation")
    window.chat_widget.conversation_id = 9

    window.new_chat_action.trigger()

    assert window.chat_widget.message_display.toPlainText() == ""
    assert window.chat_widget.conversation_id is None
    window.close()


def test_all_assistant_states_have_product_labels(qapp: object) -> None:
    runtime = _runtime()
    window = FridayMainWindow(runtime, start_background_workers=False)
    expected = {
        AssistantState.IDLE: "Ready",
        AssistantState.PROCESSING: "Processing",
        AssistantState.STREAMING: "Responding",
        AssistantState.TOOL_RUNNING: "Running tool",
        AssistantState.OFFLINE: "Offline",
        AssistantState.ERROR: "Error",
        AssistantState.LISTENING: "Listening",
        AssistantState.SPEAKING: "Speaking",
    }
    for state, text in expected.items():
        runtime.state_machine.reset()
        if state in (AssistantState.STREAMING, AssistantState.SPEAKING):
            runtime.state_machine.transition(AssistantState.PROCESSING)
        runtime.state_machine.transition(state)
        window.refresh_state()
        assert window.state_label.text() == text
    window.close()


def test_startup_health_failure_sets_offline_without_blocking_window(
    qapp: object,
) -> None:
    runtime = _runtime()

    def unavailable() -> None:
        raise OSError("offline")

    runtime.conversation_service.ollama.health_check = unavailable
    window = FridayMainWindow(runtime, start_background_workers=False)
    window.show()

    window._start_health_check()
    _wait(qapp, lambda: window._health_thread is None)

    assert window.isVisible()
    assert window.state_label.text() == "Offline"
    window.close()


def test_close_without_active_workers_is_immediate(qapp: object) -> None:
    window = FridayMainWindow(_runtime(), start_background_workers=False)
    window.show()
    qapp.processEvents()  # type: ignore[attr-defined]

    window.close()
    qapp.processEvents()  # type: ignore[attr-defined]

    assert not window.isVisible()
    assert window._shutdown_requested


def test_close_waits_for_active_chat_then_closes_automatically(
    qapp: object,
) -> None:
    gate = Event()
    conversation = FakeConversation(gate=gate)
    window = FridayMainWindow(
        _runtime(conversation=conversation),
        start_background_workers=False,
    )
    window.show()
    window.chat_widget.composer.setPlainText("hello")
    window.chat_widget.send_current_message()
    _wait(qapp, conversation.started.is_set)

    window.close()
    qapp.processEvents()  # type: ignore[attr-defined]

    assert window.isVisible()
    assert window.chat_widget.has_active_worker
    assert not window.chat_widget.composer.isEnabled()
    assert window.statusBar().currentMessage() == (
        "Finishing current background operation..."
    )

    gate.set()
    _wait(qapp, lambda: not window.isVisible())
    assert not window.chat_widget.has_active_worker


def test_close_waits_for_active_metrics_then_closes_automatically(
    qapp: object,
) -> None:
    gate = Event()
    monitor = FakeMonitor(gate)
    window = FridayMainWindow(
        _runtime(monitor=monitor),
        start_background_workers=False,
    )
    window.show()
    window.system_status.refresh_now()
    _wait(qapp, monitor.started.is_set)

    window.close()
    qapp.processEvents()  # type: ignore[attr-defined]

    assert window.isVisible()
    assert window.system_status.has_active_worker
    gate.set()
    _wait(qapp, lambda: not window.isVisible())
    assert not window.system_status.has_active_worker


def test_close_waits_for_startup_health_worker(qapp: object) -> None:
    gate = Event()
    conversation = FakeConversation(health_gate=gate)
    window = FridayMainWindow(
        _runtime(conversation=conversation),
        start_background_workers=False,
    )
    window.show()
    window._start_health_check()
    _wait(qapp, conversation.health_started.is_set)

    window.close()
    assert window.isVisible()
    assert window._health_thread is not None

    gate.set()
    _wait(qapp, lambda: not window.isVisible())
    assert window._health_thread is None
