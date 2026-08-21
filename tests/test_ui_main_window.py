from __future__ import annotations

from collections.abc import Callable
from time import monotonic, sleep
from types import SimpleNamespace

from PySide6.QtWidgets import QLabel

from friday.llm.ollama_client import OllamaResponse
from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.system.monitor import SystemMetricsSnapshot
from friday.ui.main_window import FridayMainWindow


class FakeConversation:
    def __init__(self) -> None:
        self.ollama = SimpleNamespace(health_check=lambda: object())

    def create_conversation(self, title: str | None = None) -> int:
        return 1

    def send_message(self, conversation_id: int, content: str) -> OllamaResponse:
        return OllamaResponse("ok")


class FakeMonitor:
    def get_system_metrics(self) -> SystemMetricsSnapshot:
        return SystemMetricsSnapshot(0, 0, 0, 0, 0, ())


class FakeTaskService:
    def list(self) -> list[object]:
        return []


class FakeReminderService:
    def list_upcoming(self) -> list[object]:
        return []

    def claim_due(self) -> list[object]:
        return []


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(ai_mode="local", ollama_model="test-model"),
        state_machine=AssistantStateMachine(),
        conversation_service=FakeConversation(),
        system_monitor=FakeMonitor(),
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
