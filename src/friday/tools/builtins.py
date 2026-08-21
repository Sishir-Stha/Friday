from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from friday.services.assistant_state import AssistantStateMachine
from friday.system.app_launcher import AppLauncher
from friday.system.monitor import SystemMonitor
from friday.tools.executor import ToolExecutor
from friday.tools.models import (
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolRisk,
)
from friday.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from friday.services.permission_service import PermissionService


@dataclass(frozen=True, slots=True)
class ToolRuntime:
    registry: ToolRegistry
    executor: ToolExecutor


def build_default_tool_runtime(
    *,
    state_machine: AssistantStateMachine,
    permission_service: PermissionService | None = None,
    system_monitor: SystemMonitor | None = None,
    app_launcher: AppLauncher | None = None,
) -> ToolRuntime:
    monitor = system_monitor if system_monitor is not None else SystemMonitor()
    launcher = app_launcher if app_launcher is not None else AppLauncher()
    registry = ToolRegistry()
    executor = ToolExecutor(
        registry=registry,
        permission_service=permission_service,
        state_machine=state_machine,
    )

    definitions = (
        ToolDefinition(
            name="get_system_info",
            description=(
                "Return basic local machine, operating system, CPU, and memory "
                "information."
            ),
            category="system",
            risk=ToolRisk.READ_ONLY,
        ),
        ToolDefinition(
            name="get_system_metrics",
            description=(
                "Return a current local CPU, RAM, and supported NVIDIA GPU "
                "metrics snapshot."
            ),
            category="system",
            risk=ToolRisk.READ_ONLY,
        ),
        ToolDefinition(
            name="list_allowed_apps",
            description="List canonical applications Friday is allowed to open.",
            category="applications",
            risk=ToolRisk.READ_ONLY,
        ),
        ToolDefinition(
            name="open_app",
            description="Open one application from Friday's fixed allowlist.",
            category="applications",
            risk=ToolRisk.MODIFY,
            parameters=(
                ToolParameter(
                    name="app_name",
                    description="Canonical allowed application name.",
                    parameter_type=ToolParameterType.STRING,
                    required=True,
                    allowed_values=(
                        "calculator",
                        "file_explorer",
                        "notepad",
                    ),
                ),
            ),
        ),
    )
    for definition in definitions:
        registry.register(definition)

    executor.bind(
        "get_system_info",
        lambda arguments: _get_system_info(arguments, monitor),
    )
    executor.bind(
        "get_system_metrics",
        lambda arguments: _get_system_metrics(arguments, monitor),
    )
    executor.bind(
        "list_allowed_apps",
        lambda arguments: _list_allowed_apps(arguments, launcher),
    )
    executor.bind(
        "open_app",
        lambda arguments: _open_app(arguments, launcher),
    )

    return ToolRuntime(registry=registry, executor=executor)


def _get_system_info(
    arguments: dict[str, object],
    monitor: SystemMonitor,
) -> dict[str, object]:
    _require_exact_arguments(arguments, expected=frozenset())
    return asdict(monitor.get_system_info())


def _get_system_metrics(
    arguments: dict[str, object],
    monitor: SystemMonitor,
) -> dict[str, object]:
    _require_exact_arguments(arguments, expected=frozenset())
    return asdict(monitor.get_system_metrics())


def _list_allowed_apps(
    arguments: dict[str, object],
    launcher: AppLauncher,
) -> dict[str, object]:
    _require_exact_arguments(arguments, expected=frozenset())
    return {"applications": list(launcher.list_allowed_apps())}


def _open_app(
    arguments: dict[str, object],
    launcher: AppLauncher,
) -> dict[str, object]:
    _require_exact_arguments(arguments, expected=frozenset({"app_name"}))
    return asdict(launcher.open_app(arguments["app_name"]))  # type: ignore[arg-type]


def _require_exact_arguments(
    arguments: dict[str, object],
    *,
    expected: frozenset[str],
) -> None:
    actual = frozenset(arguments)
    if actual != expected:
        expected_text = ", ".join(sorted(expected)) or "none"
        raise ValueError(f"Expected exactly these arguments: {expected_text}.")
