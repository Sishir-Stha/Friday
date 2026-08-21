from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.services.permission_service import (
    PermissionDeniedError,
    PermissionRequiredError,
    PermissionService,
)
from friday.tools import (
    DuplicateToolHandlerError,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStateError,
    ToolExecutor,
    ToolExecutorError,
    ToolHandlerNotFoundError,
    ToolNotFoundError,
    ToolRegistry,
    ToolRisk,
)


def make_tool(
    name: str = "system_info",
    *,
    risk: ToolRisk = ToolRisk.READ_ONLY,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Test metadata for {name}.",
        category="tests",
        risk=risk,
    )


def make_registry(*tools: ToolDefinition) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools or (make_tool(),):
        registry.register(tool)
    return registry


def make_executor(
    tool: ToolDefinition | None = None,
    *,
    state_machine: AssistantStateMachine | None = None,
) -> ToolExecutor:
    registry = make_registry(tool or make_tool())
    return ToolExecutor(
        registry=registry,
        state_machine=state_machine,
    )


def move_to_state(
    state_machine: AssistantStateMachine,
    target: AssistantState,
) -> None:
    paths: dict[AssistantState, tuple[AssistantState, ...]] = {
        AssistantState.IDLE: (),
        AssistantState.LISTENING: (AssistantState.LISTENING,),
        AssistantState.PROCESSING: (AssistantState.PROCESSING,),
        AssistantState.STREAMING: (
            AssistantState.PROCESSING,
            AssistantState.STREAMING,
        ),
        AssistantState.SPEAKING: (
            AssistantState.PROCESSING,
            AssistantState.SPEAKING,
        ),
        AssistantState.TOOL_RUNNING: (AssistantState.TOOL_RUNNING,),
        AssistantState.OFFLINE: (AssistantState.OFFLINE,),
        AssistantState.ERROR: (AssistantState.ERROR,),
    }
    for state in paths[target]:
        state_machine.transition(state)


def test_executor_uses_exact_injected_dependencies() -> None:
    registry = make_registry()
    permission_service = PermissionService()
    state_machine = AssistantStateMachine()

    executor = ToolExecutor(
        registry=registry,
        permission_service=permission_service,
        state_machine=state_machine,
    )

    assert executor.registry is registry
    assert executor.permission_service is permission_service
    assert executor.state_machine is state_machine


def test_default_optional_dependencies_are_independent() -> None:
    first = ToolExecutor(registry=make_registry())
    second = ToolExecutor(registry=make_registry())

    assert first.permission_service is not second.permission_service
    assert first.state_machine is not second.state_machine


def test_executor_requires_actual_registry() -> None:
    with pytest.raises(ValueError, match="ToolRegistry"):
        ToolExecutor(registry=object())  # type: ignore[arg-type]


def test_bind_valid_handler_and_has_handler() -> None:
    executor = make_executor()
    handler = lambda arguments: arguments

    executor.bind("  system_info  ", handler)

    assert executor.has_handler("system_info") is True


def test_binding_unknown_tool_preserves_registry_error() -> None:
    executor = make_executor()

    with pytest.raises(ToolNotFoundError):
        executor.bind("unknown_tool", lambda arguments: None)


def test_duplicate_handler_binding_is_rejected() -> None:
    executor = make_executor()
    original = lambda arguments: "original"
    executor.bind("system_info", original)

    with pytest.raises(
        DuplicateToolHandlerError,
        match="Tool 'system_info' already has a bound handler",
    ) as exc_info:
        executor.bind("system_info", lambda arguments: "replacement")

    assert isinstance(exc_info.value, ToolExecutorError)
    assert executor.execute("system_info").output == "original"


@pytest.mark.parametrize("handler", [None, 1, "handler", object()])
def test_bind_rejects_non_callable_handler(handler: Any) -> None:
    executor = make_executor()

    with pytest.raises(ValueError, match="handler must be callable"):
        executor.bind("system_info", handler)


def test_unbind_returns_true_then_false_without_removing_definition() -> None:
    executor = make_executor()
    executor.bind("system_info", lambda arguments: None)

    assert executor.unbind("system_info") is True
    assert executor.unbind("system_info") is False
    assert executor.has_handler("system_info") is False
    assert executor.registry.require("system_info").name == "system_info"


def test_unknown_handler_queries_return_false() -> None:
    executor = make_executor()

    assert executor.has_handler("unknown_tool") is False
    assert executor.unbind("unknown_tool") is False


def test_handler_mappings_are_independent() -> None:
    registry = make_registry()
    first = ToolExecutor(registry=registry)
    second = ToolExecutor(registry=registry)
    first.bind("system_info", lambda arguments: None)

    assert first.has_handler("system_info") is True
    assert second.has_handler("system_info") is False


def test_unknown_tool_execution_preserves_registry_error() -> None:
    executor = make_executor()

    with pytest.raises(ToolNotFoundError):
        executor.execute("unknown_tool")


def test_registered_unbound_tool_raises_handler_not_found() -> None:
    executor = make_executor()

    with pytest.raises(
        ToolHandlerNotFoundError,
        match="Tool 'system_info' has no bound handler",
    ) as exc_info:
        executor.execute("system_info")

    assert isinstance(exc_info.value, ToolExecutorError)


def test_read_only_tool_executes_without_approval() -> None:
    executor = make_executor()
    calls: list[dict[str, object]] = []
    executor.bind("system_info", lambda arguments: calls.append(arguments) or "ok")

    result = executor.execute("system_info")

    assert result == ToolExecutionResult(tool_name="system_info", output="ok")
    assert calls == [{}]


def test_modify_tool_requires_approval_before_execution() -> None:
    tool = make_tool("open_app", risk=ToolRisk.MODIFY)
    executor = make_executor(tool)
    calls: list[dict[str, object]] = []
    executor.bind("open_app", lambda arguments: calls.append(arguments))

    with pytest.raises(PermissionRequiredError):
        executor.execute("open_app")

    assert calls == []
    assert executor.state_machine.get() is AssistantState.IDLE


def test_modify_tool_executes_with_explicit_approval() -> None:
    tool = make_tool("open_app", risk=ToolRisk.MODIFY)
    executor = make_executor(tool)
    executor.bind("open_app", lambda arguments: "opened")

    result = executor.execute("open_app", user_approved=True)

    assert result.output == "opened"


@pytest.mark.parametrize("user_approved", [False, True])
def test_destructive_handler_never_executes(user_approved: bool) -> None:
    tool = make_tool("delete_file", risk=ToolRisk.DESTRUCTIVE)
    executor = make_executor(tool)
    calls: list[dict[str, object]] = []
    executor.bind("delete_file", lambda arguments: calls.append(arguments))

    with pytest.raises(PermissionDeniedError):
        executor.execute("delete_file", user_approved=user_approved)

    assert calls == []
    assert executor.state_machine.get() is AssistantState.IDLE


def test_arguments_none_passes_empty_dictionary() -> None:
    executor = make_executor()
    received: list[dict[str, object]] = []
    executor.bind("system_info", lambda arguments: received.append(arguments))

    executor.execute("system_info")

    assert received == [{}]


def test_argument_mapping_is_shallow_copied() -> None:
    executor = make_executor()
    original: dict[str, object] = {"value": 1}
    received: list[dict[str, object]] = []

    def mutating_handler(arguments: dict[str, object]) -> None:
        received.append(arguments)
        arguments["value"] = 2
        arguments["added"] = True

    executor.bind("system_info", mutating_handler)

    executor.execute("system_info", arguments=original)

    assert received[0] is not original
    assert original == {"value": 1}


@pytest.mark.parametrize("arguments", [1, "value", ["value"], object()])
def test_non_mapping_arguments_are_rejected(arguments: Any) -> None:
    executor = make_executor()
    called = False

    def handler(received: dict[str, object]) -> None:
        nonlocal called
        called = True

    executor.bind("system_info", handler)

    with pytest.raises(ValueError, match="arguments must be a mapping"):
        executor.execute("system_info", arguments=arguments)

    assert called is False


@pytest.mark.parametrize("key", [1, None, ("tuple",)])
def test_non_string_argument_keys_are_rejected(key: Any) -> None:
    executor = make_executor()
    executor.bind("system_info", lambda arguments: None)

    with pytest.raises(ValueError, match="argument keys must be strings"):
        executor.execute("system_info", arguments={key: "value"})


def test_success_result_is_immutable_and_uses_canonical_name() -> None:
    executor = make_executor()
    executor.bind("system_info", lambda arguments: {"status": "ok"})

    result = executor.execute("  system_info  ")

    assert result.tool_name == "system_info"
    assert result.output == {"status": "ok"}
    with pytest.raises(FrozenInstanceError):
        result.tool_name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "entry_state",
    [AssistantState.IDLE, AssistantState.PROCESSING, AssistantState.STREAMING],
)
def test_allowed_state_flows_restore_previous_state(
    entry_state: AssistantState,
) -> None:
    machine = AssistantStateMachine()
    move_to_state(machine, entry_state)
    executor = make_executor(state_machine=machine)
    observed_states: list[AssistantState] = []

    def handler(arguments: dict[str, object]) -> str:
        observed_states.append(machine.get())
        return "done"

    executor.bind("system_info", handler)

    assert executor.execute("system_info").output == "done"
    assert observed_states == [AssistantState.TOOL_RUNNING]
    assert machine.get() is entry_state


@pytest.mark.parametrize(
    "entry_state",
    [
        AssistantState.LISTENING,
        AssistantState.SPEAKING,
        AssistantState.TOOL_RUNNING,
        AssistantState.OFFLINE,
        AssistantState.ERROR,
    ],
)
def test_disallowed_state_rejects_without_call_or_transition(
    entry_state: AssistantState,
) -> None:
    machine = AssistantStateMachine()
    move_to_state(machine, entry_state)
    executor = make_executor(state_machine=machine)
    called = False

    def handler(arguments: dict[str, object]) -> None:
        nonlocal called
        called = True

    executor.bind("system_info", handler)

    with pytest.raises(ToolExecutionStateError, match=entry_state.value):
        executor.execute("system_info")

    assert called is False
    assert machine.get() is entry_state


def test_handler_failure_transitions_to_error_and_propagates_original() -> None:
    machine = AssistantStateMachine()
    executor = make_executor(state_machine=machine)
    error = RuntimeError("boom")

    def failing_handler(arguments: dict[str, object]) -> None:
        assert machine.get() is AssistantState.TOOL_RUNNING
        raise error

    executor.bind("system_info", failing_handler)

    with pytest.raises(RuntimeError, match="boom") as exc_info:
        executor.execute("system_info")

    assert exc_info.value is error
    assert machine.get() is AssistantState.ERROR


def test_permission_is_checked_before_entering_tool_running() -> None:
    machine = AssistantStateMachine()
    tool = make_tool("open_app", risk=ToolRisk.MODIFY)
    executor = make_executor(tool, state_machine=machine)
    executor.bind("open_app", lambda arguments: None)

    with pytest.raises(PermissionRequiredError):
        executor.execute("open_app")

    assert machine.get() is AssistantState.IDLE


def test_handler_presence_is_checked_before_permission() -> None:
    tool = make_tool("open_app", risk=ToolRisk.MODIFY)
    executor = make_executor(tool)

    with pytest.raises(ToolHandlerNotFoundError):
        executor.execute("open_app")


def test_handler_type_alias_accepts_expected_callable() -> None:
    handler: Callable[[dict[str, object]], object] = lambda arguments: arguments
    executor = make_executor()

    executor.bind("system_info", handler)

    assert executor.execute("system_info", arguments={"value": 1}).output == {
        "value": 1
    }
