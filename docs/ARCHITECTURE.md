# Friday architecture

Friday is a local-first desktop assistant. Its current local conversation path is:

```text
Windows
  -> Ollama background server
  -> http://127.0.0.1:11434
  -> Friday OllamaClient
  -> ConversationService
  -> PostgreSQL
```

`OllamaClient` owns HTTP communication with Ollama. `ConversationService` owns
conversation history and persistence. PostgreSQL is the durable source of
conversation and message data.

## Ollama health and failures

`OllamaClient.health_check()` sends `GET /api/version` with a short timeout. This
endpoint checks the server without loading a model. It is suitable for Friday
startup, status displays, diagnostics, and explicit availability checks.

The health check is deliberately not sent before every chat request. Each
`chat()` and `chat_stream()` request handles connection, timeout, HTTP, and
response-format failures independently, because Ollama can become unavailable
after an earlier health check.

Ollama failures use a small domain-specific exception hierarchy:

- `OllamaError` is the base exception.
- `OllamaUnavailableError` represents connection and transport failures.
- `OllamaTimeoutError` represents connection, read, and other HTTP timeouts.
- `OllamaResponseError` represents HTTP errors and malformed or incomplete
  Ollama responses.

The exception message is safe to surface later in a desktop UI. The endpoint,
HTTP status when present, and a bounded diagnostic detail are retained on the
exception for logging.

## Conversation modes

The non-streaming path remains `ConversationService.send_message()`. It saves the
user message once, calls `OllamaClient.chat()` with `stream=false`, then saves one
complete assistant message.

The progressive path is `ConversationService.send_message_stream()`. It saves
the user message once and returns an iterator of visible assistant text chunks.
`OllamaClient.chat_stream()` calls `/api/chat` with `stream=true`, ignores empty
content, and stops only on Ollama's completion marker.

Friday disables model reasoning/thinking output by default. Internal model
reasoning is neither displayed nor persisted. Only final assistant content is
used. This reduces unnecessary generation, latency, memory/CPU usage, UI clutter,
and database growth. Both chat modes send `think=false`; if Ollama nevertheless
returns a `thinking` field, Friday ignores it.

The service accumulates streamed content while yielding it. Only after successful
completion does it save one complete assistant message. If generation fails or
the consumer closes the iterator early, Friday does not save a partial response
as a completed assistant message. The streaming HTTP response and client are
closed by context managers.

## Runtime context

Friday has a lightweight Context Manager foundation for ephemeral runtime state.
It can track the active module, active window, selected item, current
conversation, and recent tool result. This will support contextual references in
future integrations without coupling the state container to the LLM, UI, or
operating system.

After `ConversationService` successfully persists a user message, it records the
request's `current_conversation_id` in the Context Manager. The active
conversation remains set after successful completion, during `ERROR` or
`OFFLINE`, and after explicit stream cancellation. Invalid input and nonexistent
conversations do not replace an existing valid conversation context because the
update occurs only after user-message persistence succeeds. Other context fields
are preserved.

Context Manager state is currently in-memory only and is not persisted to
PostgreSQL. It is not yet injected into LLM prompts.

## Long-term memory

Friday uses PostgreSQL for durable long-term memories. Each memory records its
type, content, importance, optional source message, and active status.
`MemoryService` provides explicit `remember`, `recall`, `get`, `forget`, and
`restore` operations. Normal recall returns active memories only, ordered by
importance and recency. Forgetting is reversible soft deactivation, not physical
deletion.

`ConversationService` reads up to five active memories through `MemoryService`
for both normal and streaming prompts. Only bounded type and content values are
included, serialized as untrusted contextual data rather than instructions.
Memory context is never persisted as a conversation message.

Automatic memory extraction and writing are not implemented. This foundation
does not use embeddings, vector search, or semantic retrieval.

## Assistant runtime state

Friday has an Assistant State Machine foundation with these states: `IDLE`,
`LISTENING`, `PROCESSING`, `STREAMING`, `SPEAKING`, `TOOL_RUNNING`, `OFFLINE`,
and `ERROR`. An explicit transition map prevents invalid runtime state changes,
while same-state updates are idempotent.

`ConversationService` now drives conversation runtime state through these flows:

```text
Normal:              IDLE -> PROCESSING -> IDLE
Streaming:           IDLE -> PROCESSING -> STREAMING -> IDLE
Ollama unavailable:  PROCESSING/STREAMING -> OFFLINE
Other failure:       PROCESSING/STREAMING -> ERROR
Stream cancellation: STREAMING -> IDLE
```

The `OFFLINE` and `ERROR` states remain observable until a later recovery layer
changes them. Assistant state is still in-memory only. Future integrations will
coordinate the desktop UI, voice, and broader Ollama connectivity.

`AssistantStateMachine` and `ContextManager` protect their in-memory state with
reentrant locks. Their synchronous APIs and immutable snapshots are therefore
safe to share among the Qt GUI, chat, monitoring, and notifier worker threads.

## Tool registry

Friday has a lightweight in-memory registry for tool metadata. An immutable
`ToolDefinition` records a stable canonical name, description, category, a
`ToolRisk` value of `READ_ONLY`, `MODIFY`, or `DESTRUCTIVE`, and immutable
parameter metadata. Parameters convert deterministically to native Ollama JSON
schemas. Registry instances are independent and support deterministic discovery,
lookup, registration, and removal; duplicate names are rejected.

The registry contains no executable callbacks. Handler bindings live separately
inside each `ToolExecutor`, preventing metadata lookup from bypassing the
permission boundary. Duplicate metadata and handler bindings are rejected.

## Tool permissions and execution

`PermissionService` is stateless and applies a deterministic risk policy to each
call. `READ_ONLY` tools are allowed, `MODIFY` tools require explicit approval for
that call, and `DESTRUCTIVE` tools are denied even with approval at this stage.
There are no stored approval preferences.

`ToolExecutor` resolves registry metadata and verifies a handler before checking
permission. It validates and copies argument mappings, enters `TOOL_RUNNING`
only after authorization, and invokes the separately bound Python handler.
Successful calls restore the prior allowed state (`IDLE`, `PROCESSING`, or
`STREAMING`). A handler failure leaves the assistant in `ERROR`, and the original
handler exception propagates unchanged.

Definitions, bindings, permission decisions, calls, and results are not persisted
to PostgreSQL.

## System monitoring

`SystemMonitor` gathers snapshots on demand; it has no sampling thread, timer,
singleton, or continuous polling. Python and `psutil` provide machine, OS, CPU,
and RAM data. CPU percentage uses a small bounded sample interval and RAM comes
from `psutil.virtual_memory()`.

NVIDIA GPU metrics use an installed `nvidia-smi` executable discovered locally
and a fixed query argument list. The command uses an argv list, a three-second
timeout, and `shell=False`. Missing NVIDIA tooling, timeouts, command failures,
or unusable output produce an empty GPU tuple without hiding valid CPU/RAM data.
Individual unavailable numeric GPU fields remain `None` instead of fabricated
zero values.

## Built-in tool runtime

`build_default_tool_runtime()` assembles one `ToolRegistry` and `ToolExecutor`
around the exact injected state machine and optional permission, monitor, and
launcher dependencies. It registers `get_system_info`, `get_system_metrics`,
and `list_allowed_apps` as `READ_ONLY`, plus `open_app` as `MODIFY`. Handlers
remain separate from metadata and return JSON-compatible values.

## Controlled application launcher

`AppLauncher` is a deliberately narrow Windows-only capability. It maps only the
canonical aliases `calculator`, `file_explorer`, and `notepad` to fixed executable
names. User or model input can select an allowlist key but cannot supply a path,
command, arguments, URL, environment expansion, or shell text. Launching uses a
fixed one-element argv list with `shell=False`.

`open_app` requires explicit per-call approval through `PermissionService`. It is
registered for direct controlled use. The desktop runtime makes it visible to
the LLM only when its GUI approval bridge is installed; headless/default service
composition continues to expose read-only tools only.

## Native Ollama tool calling

`OllamaClient.chat()` can send native tool schemas and parse validated tool calls.
An empty assistant content field is accepted only when valid tool calls exist.
Thinking remains disabled and ignored. Streaming stays visible-text-only and does
not receive tool schemas in this phase.

## Conversation tool orchestration

When an optional `ToolRuntime` is injected, non-streaming `send_message()` sends
`READ_ONLY` schemas to Ollama. If and only if a `ToolApprovalHandler` is also
injected, it additionally sends `MODIFY` schemas. Model requests are checked
again against the registry and risk before the executor runs them. Successful
execution temporarily moves `PROCESSING -> TOOL_RUNNING -> PROCESSING`; its JSON
result is sent back to Ollama as a transient tool message until a final assistant
answer is returned.

For a model-requested `MODIFY` call, the desktop bridge synchronously asks on the
Qt GUI thread while the chat worker waits. Approval is valid for that call only.
Closing the dialog or application denies the request. A denial executes nothing,
does not enter `TOOL_RUNNING`, and returns only the transient tool result
`{"status":"declined_by_user"}` to Ollama. Approval events and preferences are
not persisted. `DESTRUCTIVE` schemas are never exposed and destructive calls are
rejected before execution.

Tool-call rounds, calls per response, transient result text, and the context
snapshot are all bounded. The final successful result updates the ephemeral
`ContextManager.recent_tool_result` while preserving other context fields. Tool
planner messages, tool results, schemas, system prompts, and runtime state are
never conversation rows: PostgreSQL stores one user message and one final
assistant message only.

Tool orchestration intentionally applies only to non-streaming `send_message()`.
The existing streaming API remains available, but streaming tool orchestration
is deferred.

## Tool safety boundaries

`READ_ONLY` tools are always eligible for Ollama. The desktop runtime may expose
`MODIFY`, but every such call remains approval-gated. `DESTRUCTIVE` remains
denied, and no destructive tools exist. There are no filesystem, arbitrary
command, PowerShell, process-termination, network, or automation tools.
Production tool code accepts no arbitrary subprocess command and uses no
`shell=True` or `os.system`.

## Phase 3 desktop product layer

`build_friday_runtime()` is the composition root. Each call creates fresh
settings-backed services and exactly one shared `AssistantStateMachine` and
`ContextManager`. The tool executor and conversation service receive those exact
instances. The runtime also owns the system monitor, permission service,
`TaskService`, and `ReminderService`; importing the module performs no health
check or database query.

The PySide6 main window remains assistant-focused: a compact identity/state/model
header, the plain-text conversation and composer, and compact CPU/RAM/GPU values.
It has no task/reminder dashboard, recent-activity panel, charts, tool history, or
developer diagnostics. Tasks and reminders live in a separate organizer dialog
created only when requested from the overflow menu.

Desktop chat uses non-streaming `ConversationService.send_message()` so native
tool orchestration remains available. Conversation creation and message work run
on a `QThread`; Enter sends, Shift+Enter inserts a newline, and one chat window
cannot overlap generations. Model output is appended as plain text, never as
arbitrary rich HTML. The backend streaming API remains available for non-desktop
callers.

Startup health, three-second system metrics refreshes, organizer database work,
and 30-second reminder claims also run on workers. The GUI thread only presents
results. Metrics jobs and reminder polls use in-progress guards, so timer ticks
cannot queue unbounded work. Startup health failure shows Offline without
preventing the application from opening, and UI metrics are never persisted.

`TaskService` and `ReminderService` use fresh PostgreSQL sessions and return
immutable snapshots rather than detached ORM objects. Tasks support create,
read, list, update, start, complete, and cancel; Phase 3 never physically deletes
user tasks. Reminders are one-time only. Claiming an enabled due reminder marks
it fired and disabled in the same committed operation, preventing repeat
notifications.

While Friday runs, `ReminderNotifier` polls due reminders and uses
`QSystemTrayIcon` when available, with a transient in-app fallback otherwise.
Notification titles stay local. The desktop layer persists neither UI state,
runtime state, metrics, tool approvals, nor notification state beyond the
existing reminder fields.

Voice, wake-word interaction, speech, screen understanding, and autonomous
destructive actions are not implemented in Phase 3.

## Windows startup

Ollama is currently enabled under:

```text
Task Manager -> Startup apps -> Ollama -> Enabled
```

This starts the Ollama server with Windows. Friday itself is not yet configured
to launch automatically with Windows.

## RAM and model lifetime

The Ollama server may remain running in the background, but a running server does
not mean an LLM is permanently loaded in RAM. Ollama normally keeps a recently
used model loaded temporarily and later unloads it according to its default
keep-alive behavior.

Friday leaves that default unchanged. It is intentionally designed not to keep
the model in RAM permanently. Do not configure `keep_alive=-1` at this stage.

## Known-good local AI baseline

The verified Windows configuration is:

```text
Ollama: 0.24.0
Model: qwen2.5:3b
GPU: NVIDIA GTX 950M 4 GB
Inference: 100% GPU verified with ollama ps
Context: 4096
```

Friday stores user messages and final assistant responses only. Thinking or
reasoning traces are neither displayed nor persisted. Qwen3 was removed because
its thinking behavior was undesirable for Friday.
