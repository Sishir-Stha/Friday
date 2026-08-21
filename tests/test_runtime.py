from friday.core.runtime import build_friday_runtime


def test_runtime_uses_exact_shared_state_and_context_instances() -> None:
    runtime = build_friday_runtime()

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


def test_runtime_builder_returns_fresh_runtime_components() -> None:
    first = build_friday_runtime()
    second = build_friday_runtime()

    assert first is not second
    assert first.state_machine is not second.state_machine
    assert first.context_manager is not second.context_manager
    assert first.conversation_service is not second.conversation_service
