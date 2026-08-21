from dataclasses import dataclass

import pytest

from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.services.permission_service import (
    PermissionRequiredError,
    PermissionService,
)
from friday.system.app_launcher import AppLaunchResult
from friday.system.monitor import (
    GpuMetricsSnapshot,
    SystemInfoSnapshot,
    SystemMetricsSnapshot,
)
from friday.tools import (
    ToolParameterType,
    ToolRisk,
    build_default_tool_runtime,
)


@dataclass
class FakeSystemMonitor:
    state_machine: AssistantStateMachine | None = None
    info_calls: int = 0
    metrics_calls: int = 0

    def get_system_info(self) -> SystemInfoSnapshot:
        if self.state_machine is not None:
            assert self.state_machine.get() is AssistantState.TOOL_RUNNING
        self.info_calls += 1
        return SystemInfoSnapshot(
            hostname="friday-pc",
            os_name="Windows",
            os_release="11",
            os_version="10.0",
            architecture="AMD64",
            processor="Test CPU",
            physical_cpu_count=4,
            logical_cpu_count=8,
            total_memory_bytes=16_000,
        )

    def get_system_metrics(self) -> SystemMetricsSnapshot:
        if self.state_machine is not None:
            assert self.state_machine.get() is AssistantState.TOOL_RUNNING
        self.metrics_calls += 1
        return SystemMetricsSnapshot(
            cpu_percent=25.0,
            memory_percent=50.0,
            memory_used_bytes=8_000,
            memory_available_bytes=8_000,
            memory_total_bytes=16_000,
            gpus=(
                GpuMetricsSnapshot(
                    index=0,
                    name="Test GPU",
                    utilization_percent=40.0,
                    memory_used_mb=1000.0,
                    memory_total_mb=4000.0,
                    temperature_c=60.0,
                ),
            ),
        )


@dataclass
class FakeAppLauncher:
    open_calls: list[str]

    def list_allowed_apps(self) -> tuple[str, ...]:
        return ("calculator", "file_explorer", "notepad")

    def open_app(self, app_name: str) -> AppLaunchResult:
        self.open_calls.append(app_name)
        return AppLaunchResult(app_name=app_name, pid=1234)


def build_runtime(
    *,
    machine: AssistantStateMachine | None = None,
    monitor: FakeSystemMonitor | None = None,
    launcher: FakeAppLauncher | None = None,
    permission_service: PermissionService | None = None,
):
    state_machine = machine if machine is not None else AssistantStateMachine()
    return build_default_tool_runtime(
        state_machine=state_machine,
        permission_service=permission_service,
        system_monitor=monitor if monitor is not None else FakeSystemMonitor(),
        app_launcher=(
            launcher if launcher is not None else FakeAppLauncher(open_calls=[])
        ),
    )


def test_runtime_registers_exact_builtin_definitions() -> None:
    runtime = build_runtime()

    assert [tool.name for tool in runtime.registry.list_tools()] == [
        "get_system_info",
        "get_system_metrics",
        "list_allowed_apps",
        "open_app",
    ]
    expected = {
        "get_system_info": ("system", ToolRisk.READ_ONLY),
        "get_system_metrics": ("system", ToolRisk.READ_ONLY),
        "list_allowed_apps": ("applications", ToolRisk.READ_ONLY),
        "open_app": ("applications", ToolRisk.MODIFY),
    }
    assert {
        tool.name: (tool.category, tool.risk)
        for tool in runtime.registry.list_tools()
    } == expected


def test_open_app_definition_has_bounded_parameter_schema() -> None:
    runtime = build_runtime()
    tool = runtime.registry.require("open_app")

    assert len(tool.parameters) == 1
    parameter = tool.parameters[0]
    assert parameter.name == "app_name"
    assert parameter.parameter_type is ToolParameterType.STRING
    assert parameter.required is True
    assert parameter.allowed_values == (
        "calculator",
        "file_explorer",
        "notepad",
    )


def test_all_builtin_handlers_are_bound() -> None:
    runtime = build_runtime()

    assert all(
        runtime.executor.has_handler(tool.name)
        for tool in runtime.registry.list_tools()
    )


def test_runtime_uses_exact_injected_dependencies() -> None:
    machine = AssistantStateMachine()
    permission_service = PermissionService()
    monitor = FakeSystemMonitor()
    launcher = FakeAppLauncher(open_calls=[])

    runtime = build_default_tool_runtime(
        state_machine=machine,
        permission_service=permission_service,
        system_monitor=monitor,
        app_launcher=launcher,
    )

    assert runtime.executor.state_machine is machine
    assert runtime.executor.permission_service is permission_service
    runtime.executor.execute("get_system_info")
    assert monitor.info_calls == 1
    runtime.executor.execute("list_allowed_apps")


def test_runtime_builds_independent_registry_executor_and_defaults() -> None:
    first = build_default_tool_runtime(state_machine=AssistantStateMachine())
    second = build_default_tool_runtime(state_machine=AssistantStateMachine())

    assert first.registry is not second.registry
    assert first.executor is not second.executor
    assert first.executor.permission_service is not second.executor.permission_service


def test_get_system_info_handler_returns_json_compatible_dictionary() -> None:
    monitor = FakeSystemMonitor()
    runtime = build_runtime(monitor=monitor)

    result = runtime.executor.execute("get_system_info")

    assert result.output == {
        "hostname": "friday-pc",
        "os_name": "Windows",
        "os_release": "11",
        "os_version": "10.0",
        "architecture": "AMD64",
        "processor": "Test CPU",
        "physical_cpu_count": 4,
        "logical_cpu_count": 8,
        "total_memory_bytes": 16_000,
    }


def test_get_system_metrics_handler_returns_nested_dictionary() -> None:
    monitor = FakeSystemMonitor()
    runtime = build_runtime(monitor=monitor)

    result = runtime.executor.execute("get_system_metrics")

    assert result.output["cpu_percent"] == 25.0  # type: ignore[index]
    assert result.output["memory_percent"] == 50.0  # type: ignore[index]
    assert result.output["gpus"][0]["name"] == "Test GPU"  # type: ignore[index]


def test_list_allowed_apps_handler_returns_list() -> None:
    runtime = build_runtime()

    result = runtime.executor.execute("list_allowed_apps")

    assert result.output == {
        "applications": ["calculator", "file_explorer", "notepad"]
    }


def test_read_only_execution_enters_tool_running_and_restores_state() -> None:
    machine = AssistantStateMachine()
    monitor = FakeSystemMonitor(state_machine=machine)
    runtime = build_runtime(machine=machine, monitor=monitor)

    runtime.executor.execute("get_system_info")

    assert machine.get() is AssistantState.IDLE


def test_open_app_requires_approval_before_handler() -> None:
    launcher = FakeAppLauncher(open_calls=[])
    runtime = build_runtime(launcher=launcher)

    with pytest.raises(PermissionRequiredError):
        runtime.executor.execute(
            "open_app",
            arguments={"app_name": "notepad"},
        )

    assert launcher.open_calls == []


def test_open_app_executes_with_explicit_approval() -> None:
    launcher = FakeAppLauncher(open_calls=[])
    runtime = build_runtime(launcher=launcher)

    result = runtime.executor.execute(
        "open_app",
        arguments={"app_name": "notepad"},
        user_approved=True,
    )

    assert result.output == {"app_name": "notepad", "pid": 1234}
    assert launcher.open_calls == ["notepad"]


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_system_info", {"unexpected": True}),
        ("get_system_metrics", {"unexpected": True}),
        ("list_allowed_apps", {"unexpected": True}),
        ("open_app", {}),
        ("open_app", {"app_name": "notepad", "unexpected": True}),
    ],
)
def test_builtin_handlers_require_exact_argument_keys(
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    runtime = build_runtime()

    with pytest.raises(ValueError, match="Expected exactly"):
        runtime.executor.execute(
            tool_name,
            arguments=arguments,
            user_approved=True,
        )
