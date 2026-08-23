from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from time import monotonic, sleep

from friday.tools.models import ToolDefinition, ToolRisk
from friday.ui.approval_bridge import ToolApprovalBridge


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="open_app",
        description="Open an allowlisted application.",
        category="applications",
        risk=ToolRisk.MODIFY,
    )


def _wait(qapp: object, condition: Callable[[], bool]) -> None:
    deadline = monotonic() + 2
    while not condition() and monotonic() < deadline:
        qapp.processEvents()  # type: ignore[attr-defined]
        sleep(0.005)
    assert condition()


def test_worker_request_is_resolved_on_gui_thread(qapp: object) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    result: list[bool] = []
    bridge = ToolApprovalBridge(
        prompt=lambda tool, arguments: calls.append(
            (tool.name, dict(arguments))
        )
        or True
    )
    thread = Thread(
        target=lambda: result.append(bridge(_tool(), {"app_name": "notepad"}))
    )

    thread.start()
    _wait(qapp, lambda: bool(result))
    thread.join(timeout=1)

    assert result == [True]
    assert calls == [("open_app", {"app_name": "notepad"})]


def test_deny_and_closed_prompt_return_false(qapp: object) -> None:
    bridge = ToolApprovalBridge(prompt=lambda tool, arguments: False)

    assert bridge(_tool(), {"app_name": "notepad"}) is False


def test_approval_is_not_persistent(qapp: object) -> None:
    answers = iter((True, False))
    bridge = ToolApprovalBridge(prompt=lambda tool, arguments: next(answers))

    assert bridge(_tool(), {}) is True
    assert bridge(_tool(), {}) is False


def test_shutdown_releases_waiting_request_as_denied(qapp: object) -> None:
    result: list[bool] = []
    bridge = ToolApprovalBridge(prompt=lambda tool, arguments: True)
    thread = Thread(target=lambda: result.append(bridge(_tool(), {})))

    thread.start()
    bridge.shutdown()
    thread.join(timeout=1)

    assert result == [False]
