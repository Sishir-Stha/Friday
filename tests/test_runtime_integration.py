import pytest

from friday.core.runtime import build_friday_runtime

pytestmark = pytest.mark.integration


def test_desktop_runtime_composition() -> None:
    runtime = build_friday_runtime(tool_approval_handler=lambda tool, arguments: False)

    assert runtime.conversation_service.state_machine is runtime.state_machine
    assert runtime.tool_runtime.executor.state_machine is runtime.state_machine
    assert runtime.conversation_service.context_manager is runtime.context_manager
    assert runtime.task_service is not None
    assert runtime.reminder_service is not None
    assert runtime.system_monitor is not None
    assert {tool.name for tool in runtime.tool_runtime.registry.list_tools()} == {
        "get_system_info",
        "get_system_metrics",
        "list_allowed_apps",
        "open_app",
    }
