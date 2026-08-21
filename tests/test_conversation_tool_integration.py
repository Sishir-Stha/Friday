import json
from collections.abc import Iterator
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from friday.llm.conversation_service import (
    MAX_RECENT_TOOL_RESULT_CHARS,
    MAX_TOOL_CALLS_PER_ROUND,
    MAX_TOOL_RESULT_CHARS,
    MAX_TOOL_ROUNDS,
    ConversationService,
    ToolOrchestrationError,
    ToolOrchestrationLimitError,
)
from friday.llm.ollama_client import (
    ChatMessage,
    OllamaResponse,
    OllamaToolCall,
    OllamaUnavailableError,
)
from friday.llm.prompts import FRIDAY_SYSTEM_PROMPT
from friday.services.assistant_state import AssistantState, AssistantStateMachine
from friday.services.context_manager import ContextManager
from friday.tools.builtins import ToolRuntime
from friday.tools.executor import ToolExecutor
from friday.tools.models import ToolDefinition, ToolRisk
from friday.tools.registry import ToolRegistry


class RecordingConversationService(ConversationService):
    def __init__(
        self,
        *,
        ollama: Any,
        state_machine: AssistantStateMachine,
        context_manager: ContextManager | None = None,
        tool_runtime: ToolRuntime | None = None,
        tool_approval_handler: Any | None = None,
        system_content: str = FRIDAY_SYSTEM_PROMPT,
    ) -> None:
        super().__init__(
            ollama=ollama,
            settings=SimpleNamespace(ai_mode="local", ollama_model="test-model"),
            state_machine=state_machine,
            context_manager=context_manager,
            tool_runtime=tool_runtime,
            tool_approval_handler=tool_approval_handler,
        )
        self.persisted_messages: list[tuple[str, str]] = []
        self.system_content = system_content

    def _persist_user_message(self, conversation_id: int, content: str) -> None:
        self.persisted_messages.append(("user", content))

    def _persist_assistant_message(
        self,
        conversation_id: int,
        content: str,
    ) -> None:
        self.persisted_messages.append(("assistant", content))

    def _build_llm_messages(self, conversation_id: int) -> list[ChatMessage]:
        return [
            {"role": "system", "content": self.system_content},
            *[
                {"role": role, "content": content}
                for role, content in self.persisted_messages
            ],
        ]


class ScriptedOllama:
    def __init__(self, *responses: OllamaResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[ChatMessage], list[dict[str, object]]]] = []

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, object]],
    ) -> OllamaResponse:
        self.calls.append((deepcopy(messages), deepcopy(tools)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class LegacyOllama:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] | None = None

    def chat(self, messages: list[ChatMessage]) -> OllamaResponse:
        self.messages = deepcopy(messages)
        return OllamaResponse(content="legacy answer")


class StreamingOnlyOllama:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] | None = None

    def chat_stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        self.messages = deepcopy(messages)
        yield "streamed"
        yield " answer"


def tool_call(name: str, arguments: dict[str, object] | None = None) -> OllamaResponse:
    return OllamaResponse(
        content="",
        tool_calls=(OllamaToolCall(name=name, arguments=arguments or {}),),
    )


def make_runtime(
    machine: AssistantStateMachine,
    *,
    read_handler: Any | None = None,
    modify_handler: Any | None = None,
) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_system_metrics",
            description="Return current metrics.",
            category="system",
            risk=ToolRisk.READ_ONLY,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_allowed_apps",
            description="List allowed applications.",
            category="applications",
            risk=ToolRisk.READ_ONLY,
        )
    )
    registry.register(
        ToolDefinition(
            name="open_app",
            description="Open an allowed application.",
            category="applications",
            risk=ToolRisk.MODIFY,
        )
    )
    executor = ToolExecutor(registry=registry, state_machine=machine)
    executor.bind(
        "get_system_metrics",
        read_handler if read_handler is not None else lambda arguments: {"cpu": 12},
    )
    executor.bind("list_allowed_apps", lambda arguments: {"apps": ["notepad"]})
    executor.bind(
        "open_app",
        modify_handler if modify_handler is not None else lambda arguments: {"pid": 1},
    )
    return ToolRuntime(registry=registry, executor=executor)


def make_service(
    ollama: Any,
    *,
    machine: AssistantStateMachine | None = None,
    context: ContextManager | None = None,
    runtime: ToolRuntime | None = None,
    tool_approval_handler: Any | None = None,
    system_content: str = FRIDAY_SYSTEM_PROMPT,
) -> RecordingConversationService:
    machine = machine if machine is not None else AssistantStateMachine()
    return RecordingConversationService(
        ollama=ollama,
        state_machine=machine,
        context_manager=context,
        tool_runtime=runtime,
        tool_approval_handler=tool_approval_handler,
        system_content=system_content,
    )


def test_tool_runtime_uses_exact_injected_instance() -> None:
    machine = AssistantStateMachine()
    runtime = make_runtime(machine)

    service = make_service(ScriptedOllama(OllamaResponse("ok")), machine=machine, runtime=runtime)

    assert service.tool_runtime is runtime


def test_mismatched_tool_runtime_state_machine_is_rejected() -> None:
    runtime = make_runtime(AssistantStateMachine())

    with pytest.raises(ValueError, match="share one state machine"):
        make_service(
            ScriptedOllama(OllamaResponse("unused")),
            machine=AssistantStateMachine(),
            runtime=runtime,
        )


def test_no_runtime_preserves_legacy_ollama_call_and_persistence() -> None:
    ollama = LegacyOllama()
    service = make_service(ollama)

    response = service.send_message(1, "hello")

    assert response.content == "legacy answer"
    assert ollama.messages == [
        {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
    ]
    assert service.persisted_messages == [
        ("user", "hello"),
        ("assistant", "legacy answer"),
    ]


def test_runtime_sends_only_read_only_schemas_and_persists_no_tool_data() -> None:
    machine = AssistantStateMachine()
    runtime = make_runtime(machine)
    ollama = ScriptedOllama(OllamaResponse(content="direct answer"))
    service = make_service(ollama, machine=machine, runtime=runtime)

    response = service.send_message(1, "hello")

    schema_names = [call["function"]["name"] for call in ollama.calls[0][1]]
    assert schema_names == ["get_system_metrics", "list_allowed_apps"]
    assert "open_app" not in schema_names
    assert response.content == "direct answer"
    assert service.persisted_messages == [
        ("user", "hello"),
        ("assistant", "direct answer"),
    ]


def test_single_tool_call_observes_states_updates_context_and_is_transient() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    context.update(
        active_module="system",
        active_window="Task Manager",
        selected_item="cpu",
    )
    handler_states: list[AssistantState] = []

    def handler(arguments: dict[str, object]) -> dict[str, int]:
        assert arguments == {}
        handler_states.append(machine.get())
        return {"cpu": 37}

    runtime = make_runtime(machine, read_handler=handler)
    original_execute = runtime.executor.execute
    executor_entry_states: list[AssistantState] = []
    executor_exit_states: list[AssistantState] = []

    def recording_execute(*args: Any, **kwargs: Any) -> Any:
        executor_entry_states.append(machine.get())
        result = original_execute(*args, **kwargs)
        executor_exit_states.append(machine.get())
        return result

    runtime.executor.execute = recording_execute  # type: ignore[method-assign]
    ollama = ScriptedOllama(
        tool_call("get_system_metrics"),
        OllamaResponse(content="CPU is 37%."),
    )
    service = make_service(
        ollama,
        machine=machine,
        context=context,
        runtime=runtime,
    )

    response = service.send_message(9, "CPU?")

    assert response.content == "CPU is 37%."
    assert executor_entry_states == [AssistantState.PROCESSING]
    assert handler_states == [AssistantState.TOOL_RUNNING]
    assert executor_exit_states == [AssistantState.PROCESSING]
    assert machine.get() is AssistantState.IDLE
    second_messages = ollama.calls[1][0]
    assert second_messages[-2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_system_metrics", "arguments": {}}}
        ],
    }
    assert second_messages[-1] == {
        "role": "tool",
        "tool_name": "get_system_metrics",
        "content": '{"cpu": 37}',
    }
    assert service.persisted_messages == [
        ("user", "CPU?"),
        ("assistant", "CPU is 37%."),
    ]
    snapshot = context.get()
    assert snapshot.current_conversation_id == 9
    assert snapshot.active_module == "system"
    assert snapshot.active_window == "Task Manager"
    assert snapshot.selected_item == "cpu"
    assert json.loads(snapshot.recent_tool_result or "") == {
        "output": {"cpu": 37},
        "tool": "get_system_metrics",
    }


def test_multiple_sequential_read_only_calls_keep_only_final_recent_result() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    runtime = make_runtime(machine)
    ollama = ScriptedOllama(
        tool_call("get_system_metrics"),
        tool_call("list_allowed_apps"),
        OllamaResponse(content="done"),
    )
    service = make_service(
        ollama,
        machine=machine,
        context=context,
        runtime=runtime,
    )

    service.send_message(1, "check")

    assert len(ollama.calls) == 3
    assert [message["role"] for message in ollama.calls[2][0][-4:]] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    recent = json.loads(context.get().recent_tool_result or "")
    assert recent == {
        "output": {"apps": ["notepad"]},
        "tool": "list_allowed_apps",
    }
    assert service.persisted_messages == [("user", "check"), ("assistant", "done")]


def test_excessive_call_batch_executes_none_and_persists_no_assistant() -> None:
    machine = AssistantStateMachine()
    executions: list[dict[str, object]] = []
    runtime = make_runtime(machine, read_handler=lambda arguments: executions.append(arguments))
    calls = tuple(
        OllamaToolCall(name="get_system_metrics", arguments={})
        for _ in range(MAX_TOOL_CALLS_PER_ROUND + 1)
    )
    service = make_service(
        ScriptedOllama(OllamaResponse(content="", tool_calls=calls)),
        machine=machine,
        runtime=runtime,
    )

    with pytest.raises(ToolOrchestrationLimitError, match="too many tools"):
        service.send_message(1, "check")

    assert executions == []
    assert machine.get() is AssistantState.ERROR
    assert service.persisted_messages == [("user", "check")]


def test_tool_round_limit_stops_before_executing_excess_round() -> None:
    machine = AssistantStateMachine()
    executions: list[int] = []

    def handler(arguments: dict[str, object]) -> dict[str, int]:
        executions.append(1)
        return {"count": len(executions)}

    runtime = make_runtime(machine, read_handler=handler)
    service = make_service(
        ScriptedOllama(
            *(tool_call("get_system_metrics") for _ in range(MAX_TOOL_ROUNDS + 1))
        ),
        machine=machine,
        runtime=runtime,
    )

    with pytest.raises(ToolOrchestrationLimitError, match="round tool limit"):
        service.send_message(1, "loop")

    assert len(executions) == MAX_TOOL_ROUNDS
    assert service.persisted_messages == [("user", "loop")]


def test_unknown_and_modify_tools_are_rejected_before_any_handler() -> None:
    for name, match in (
        ("not_registered", "unknown tool"),
        ("open_app", "non-read-only"),
    ):
        machine = AssistantStateMachine()
        modify_calls: list[dict[str, object]] = []
        runtime = make_runtime(
            machine,
            modify_handler=lambda arguments, calls=modify_calls: calls.append(
                arguments
            ),
        )
        service = make_service(
            ScriptedOllama(tool_call(name)),
            machine=machine,
            runtime=runtime,
        )

        with pytest.raises(ToolOrchestrationError, match=match):
            service.send_message(1, "unsafe")

        assert modify_calls == []
        assert machine.get() is AssistantState.ERROR
        assert service.persisted_messages == [("user", "unsafe")]


def test_approval_handler_exposes_modify_and_approved_call_executes_once() -> None:
    machine = AssistantStateMachine()
    approvals: list[tuple[ToolDefinition, dict[str, object]]] = []
    executions: list[dict[str, object]] = []
    runtime = make_runtime(
        machine,
        modify_handler=lambda arguments: executions.append(arguments) or {"pid": 7},
    )

    def approve(tool: ToolDefinition, arguments: Any) -> bool:
        approvals.append((tool, dict(arguments)))
        with pytest.raises(TypeError):
            arguments["changed"] = True
        return True

    ollama = ScriptedOllama(
        tool_call("open_app", {"app_name": "notepad"}),
        OllamaResponse(content="Opened Notepad."),
    )
    service = make_service(
        ollama,
        machine=machine,
        runtime=runtime,
        tool_approval_handler=approve,
    )

    response = service.send_message(1, "Open Notepad")

    schema_names = [tool["function"]["name"] for tool in ollama.calls[0][1]]
    assert schema_names == ["get_system_metrics", "list_allowed_apps", "open_app"]
    assert approvals == [(runtime.registry.require("open_app"), {"app_name": "notepad"})]
    assert executions == [{"app_name": "notepad"}]
    assert response.content == "Opened Notepad."


def test_declined_modify_is_transient_and_never_enters_tool_running() -> None:
    machine = AssistantStateMachine()
    observed_states: list[AssistantState] = []
    executions: list[dict[str, object]] = []
    runtime = make_runtime(
        machine,
        modify_handler=lambda arguments: executions.append(arguments),
    )

    def decline(tool: ToolDefinition, arguments: Any) -> bool:
        observed_states.append(machine.get())
        return False

    ollama = ScriptedOllama(
        tool_call("open_app", {"app_name": "notepad"}),
        OllamaResponse(content="Okay, I did not open it."),
    )
    service = make_service(
        ollama,
        machine=machine,
        runtime=runtime,
        tool_approval_handler=decline,
    )

    response = service.send_message(1, "Open Notepad")

    assert observed_states == [AssistantState.PROCESSING]
    assert executions == []
    assert ollama.calls[1][0][-1] == {
        "role": "tool",
        "tool_name": "open_app",
        "content": '{"status":"declined_by_user"}',
    }
    assert response.content == "Okay, I did not open it."
    assert service.persisted_messages == [
        ("user", "Open Notepad"),
        ("assistant", "Okay, I did not open it."),
    ]


def test_destructive_tool_is_never_exposed_or_executed() -> None:
    machine = AssistantStateMachine()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="erase_everything",
            description="A destructive test tool.",
            category="system",
            risk=ToolRisk.DESTRUCTIVE,
        )
    )
    executions: list[dict[str, object]] = []
    executor = ToolExecutor(registry=registry, state_machine=machine)
    executor.bind("erase_everything", lambda arguments: executions.append(arguments))
    runtime = ToolRuntime(registry=registry, executor=executor)
    ollama = ScriptedOllama(tool_call("erase_everything"))
    service = make_service(
        ollama,
        machine=machine,
        runtime=runtime,
        tool_approval_handler=lambda tool, arguments: True,
    )

    with pytest.raises(ToolOrchestrationError, match="destructive"):
        service.send_message(1, "unsafe")

    assert ollama.calls[0][1] == []
    assert executions == []


def test_handler_exception_is_not_wrapped_and_leaves_error() -> None:
    machine = AssistantStateMachine()
    original = LookupError("sensor failed")

    def handler(arguments: dict[str, object]) -> object:
        raise original

    runtime = make_runtime(machine, read_handler=handler)
    service = make_service(
        ScriptedOllama(tool_call("get_system_metrics")),
        machine=machine,
        runtime=runtime,
    )

    with pytest.raises(LookupError, match="sensor failed") as exc_info:
        service.send_message(1, "check")

    assert exc_info.value is original
    assert machine.get() is AssistantState.ERROR
    assert service.persisted_messages == [("user", "check")]


def test_ollama_unavailable_after_tool_leaves_recent_context_and_offline() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    runtime = make_runtime(machine)
    unavailable = OllamaUnavailableError("offline", endpoint="http://ollama/api/chat")
    service = make_service(
        ScriptedOllama(tool_call("get_system_metrics"), unavailable),
        machine=machine,
        context=context,
        runtime=runtime,
    )

    with pytest.raises(OllamaUnavailableError) as exc_info:
        service.send_message(2, "check")

    assert exc_info.value is unavailable
    assert machine.get() is AssistantState.OFFLINE
    assert "get_system_metrics" in (context.get().recent_tool_result or "")
    assert service.persisted_messages == [("user", "check")]


def test_memory_system_message_stays_first_and_base_prompt_appears_once() -> None:
    machine = AssistantStateMachine()
    runtime = make_runtime(machine)
    system_content = f"{FRIDAY_SYSTEM_PROMPT}\n\nLong-term memory context:\n- remembered"
    ollama = ScriptedOllama(
        tool_call("get_system_metrics"),
        OllamaResponse(content="done"),
    )
    service = make_service(
        ollama,
        machine=machine,
        runtime=runtime,
        system_content=system_content,
    )

    service.send_message(1, "check")

    for messages, _tools in ollama.calls:
        assert messages[0] == {"role": "system", "content": system_content}
        assert messages[0]["content"].count(FRIDAY_SYSTEM_PROMPT) == 1


def test_tool_output_serialization_is_deterministic_bounded_and_safe() -> None:
    machine = AssistantStateMachine()
    context = ContextManager()
    runtime = make_runtime(
        machine,
        read_handler=lambda arguments: {"z": "x" * 5000, "a": 1},
    )
    ollama = ScriptedOllama(
        tool_call("get_system_metrics"),
        OllamaResponse(content="done"),
    )
    service = make_service(
        ollama,
        machine=machine,
        context=context,
        runtime=runtime,
    )

    service.send_message(1, "check")

    serialized = ollama.calls[1][0][-1]["content"]
    assert isinstance(serialized, str)
    assert serialized.startswith('{"a": 1, "z": "')
    assert serialized.endswith("...")
    assert len(serialized) == MAX_TOOL_RESULT_CHARS
    recent = context.get().recent_tool_result
    assert recent is not None and len(recent) == MAX_RECENT_TOOL_RESULT_CHARS
    assert recent.endswith("...")


def test_unsupported_tool_output_raises_clear_orchestration_error() -> None:
    machine = AssistantStateMachine()
    runtime = make_runtime(machine, read_handler=lambda arguments: object())
    service = make_service(
        ScriptedOllama(tool_call("get_system_metrics")),
        machine=machine,
        runtime=runtime,
    )

    with pytest.raises(ToolOrchestrationError, match="not JSON-compatible"):
        service.send_message(1, "check")

    assert machine.get() is AssistantState.ERROR
    assert service.persisted_messages == [("user", "check")]


def test_runtime_does_not_enable_tool_schemas_or_execution_for_streaming() -> None:
    machine = AssistantStateMachine()
    runtime = make_runtime(machine)
    ollama = StreamingOnlyOllama()
    service = make_service(ollama, machine=machine, runtime=runtime)

    chunks = list(service.send_message_stream(1, "hello"))

    assert chunks == ["streamed", " answer"]
    assert ollama.messages == [
        {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
    ]
    assert service.persisted_messages == [
        ("user", "hello"),
        ("assistant", "streamed answer"),
    ]
    assert machine.get() is AssistantState.IDLE
