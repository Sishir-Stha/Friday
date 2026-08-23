# Friday Development Roadmap

## Phase 1 — Core AI Foundation

Status: COMPLETE

- PostgreSQL persistence
- Local Ollama conversation
- Streaming responses
- Assistant state machine
- Context manager
- Long-term memory
- Bounded memory prompt retrieval
- Real memory/Ollama integration
- Tool Registry metadata foundation

## Phase 2 — Safe Tool Execution

Status: COMPLETE

- [x] PermissionService
- [x] ToolExecutor
- [x] Safe execution state handling
- [x] Read-only system tools
- [x] CPU/RAM/GPU monitoring
- [x] Controlled local tools
- [x] Native read-only LLM tool orchestration

Phase 2 native LLM tool orchestration is non-streaming and exposes READ_ONLY
tools only. MODIFY tools require explicit caller approval and are not
model-visible. Streaming tool orchestration and UI-mediated approval are
deferred, and destructive tools are not implemented.

## Phase 3 — Desktop Assistant Product

Status: COMPLETE

- [x] Desktop runtime composition
- [x] Minimal assistant-focused main UI
- [x] Responsive chat workflow
- [x] Live assistant/system status
- [x] UI-mediated MODIFY approval
- [x] Task management
- [x] Reminder management
- [x] Reminder notifications
- [x] Desktop smoke/integration verification

Phase 3 provides the desktop product layer with manual per-call MODIFY approval.
Chat runs asynchronously so the Qt UI remains responsive. Tasks and one-time
reminders are stored in PostgreSQL and managed in an on-demand organizer window.
Voice, wake-word interaction, screen understanding, autonomous destructive
actions, and advanced automation remain outside Phase 3.

## Phase 4 — Voice & Advanced Interaction

Status: PLANNED
