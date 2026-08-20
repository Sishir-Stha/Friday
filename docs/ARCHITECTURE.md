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
coordinate the desktop UI, voice, tool execution, and broader Ollama connectivity;
those integrations do not exist yet.

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
