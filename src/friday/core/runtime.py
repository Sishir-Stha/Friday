from dataclasses import dataclass

from friday.core.config import Settings, get_settings
from friday.llm.conversation_service import ConversationService, ToolApprovalHandler
from friday.services.assistant_state import AssistantStateMachine
from friday.services.context_manager import ContextManager
from friday.services.permission_service import PermissionService
from friday.services.reminder_service import ReminderService
from friday.services.task_service import TaskService
from friday.system.monitor import SystemMonitor
from friday.tools.builtins import ToolRuntime, build_default_tool_runtime


@dataclass(frozen=True, slots=True)
class FridayRuntime:
    settings: Settings
    state_machine: AssistantStateMachine
    context_manager: ContextManager
    permission_service: PermissionService
    system_monitor: SystemMonitor
    tool_runtime: ToolRuntime
    conversation_service: ConversationService
    task_service: TaskService
    reminder_service: ReminderService


def build_friday_runtime(
    *,
    tool_approval_handler: ToolApprovalHandler | None = None,
) -> FridayRuntime:
    settings = get_settings()
    state_machine = AssistantStateMachine()
    context_manager = ContextManager()
    permission_service = PermissionService()
    system_monitor = SystemMonitor()
    tool_runtime = build_default_tool_runtime(
        state_machine=state_machine,
        permission_service=permission_service,
        system_monitor=system_monitor,
    )
    conversation_service = ConversationService(
        settings=settings,
        state_machine=state_machine,
        context_manager=context_manager,
        tool_runtime=tool_runtime,
        tool_approval_handler=tool_approval_handler,
    )

    return FridayRuntime(
        settings=settings,
        state_machine=state_machine,
        context_manager=context_manager,
        permission_service=permission_service,
        system_monitor=system_monitor,
        tool_runtime=tool_runtime,
        conversation_service=conversation_service,
        task_service=TaskService(),
        reminder_service=ReminderService(),
    )
