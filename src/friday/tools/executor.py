from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from friday.services.permission_service import PermissionService

ToolHandler = Callable[[dict[str, object]], object]

_ALLOWED_ENTRY_STATES = frozenset(
    {
        AssistantState.IDLE,
        AssistantState.PROCESSING,
        AssistantState.STREAMING,
    }
)


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    tool_name: str
    output: object


class ToolExecutorError(Exception):
    """Base exception for tool executor failures."""


class DuplicateToolHandlerError(ToolExecutorError):
    """Raised when a tool already has a bound handler."""


class ToolHandlerNotFoundError(ToolExecutorError):
    """Raised when a registered tool has no bound handler."""


class ToolExecutionStateError(ToolExecutorError):
    """Raised when assistant state does not permit tool execution."""


class ToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        permission_service: PermissionService | None = None,
        state_machine: AssistantStateMachine | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise ValueError("registry must be a ToolRegistry")  # noqa: TRY004

        if permission_service is None:
            from friday.services.permission_service import PermissionService

            permission_service = PermissionService()

        self.registry = registry
        self.permission_service = permission_service
        self.state_machine = (
            state_machine
            if state_machine is not None
            else AssistantStateMachine()
        )
        self._handlers: dict[str, ToolHandler] = {}

    def bind(self, name: str, handler: ToolHandler) -> None:
        tool = self.registry.require(name)

        if not callable(handler):
            raise ValueError("handler must be callable")  # noqa: TRY004

        if tool.name in self._handlers:
            raise DuplicateToolHandlerError(
                f"Tool '{tool.name}' already has a bound handler."
            )

        self._handlers[tool.name] = handler

    def unbind(self, name: str) -> bool:
        tool = self.registry.get(name)
        if tool is None:
            return False
        return self._handlers.pop(tool.name, None) is not None

    def has_handler(self, name: str) -> bool:
        tool = self.registry.get(name)
        return tool is not None and tool.name in self._handlers

    def execute(
        self,
        name: str,
        *,
        arguments: Mapping[str, object] | None = None,
        user_approved: bool = False,
    ) -> ToolExecutionResult:
        tool = self.registry.require(name)
        handler = self._handlers.get(tool.name)

        if handler is None:
            raise ToolHandlerNotFoundError(
                f"Tool '{tool.name}' has no bound handler."
            )

        copied_arguments = _copy_arguments(arguments)
        self.permission_service.authorize(
            tool,
            user_approved=user_approved,
        )

        previous_state = self.state_machine.get()
        if previous_state not in _ALLOWED_ENTRY_STATES:
            raise ToolExecutionStateError(
                f"Tool '{tool.name}' cannot execute while assistant state is "
                f"'{previous_state.value}'."
            )

        self.state_machine.transition(AssistantState.TOOL_RUNNING)

        try:
            output = handler(copied_arguments)
        except Exception:
            self.state_machine.transition(AssistantState.ERROR)
            raise

        self.state_machine.transition(previous_state)
        return ToolExecutionResult(
            tool_name=tool.name,
            output=output,
        )


def _copy_arguments(
    arguments: Mapping[str, object] | None,
) -> dict[str, object]:
    if arguments is None:
        return {}

    if not isinstance(arguments, Mapping):
        raise ValueError("arguments must be a mapping")  # noqa: TRY004

    if any(not isinstance(key, str) for key in arguments):
        raise ValueError("argument keys must be strings")

    return dict(arguments)
