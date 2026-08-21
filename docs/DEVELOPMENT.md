# Development

Friday uses Python 3.12 and is managed with `uv`. PostgreSQL provides persistent
storage, and the local Ollama API is expected at `http://127.0.0.1:11434`.

## Fast automated tests

Run unit tests that do not require PostgreSQL, a running Ollama server, or model
generation:

```powershell
uv run pytest -m "not integration" -v
```

Expected result: the mocked Ollama health, failure, response-parsing, streaming,
and one-time persistence tests pass without contacting a real Ollama server.

Tests marked `integration` intentionally require local services. The marker does
not skip or disable them; it lets developers choose the correct test group.

## Ollama troubleshooting

Check the server version without loading a model:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/version
```

Expected result: PowerShell returns Ollama version information.

List locally installed models:

```powershell
ollama list
```

Expected result: Ollama prints the installed model list.

Inspect models that are currently loaded:

```powershell
ollama ps
```

Expected result: Ollama prints loaded/running models, or an empty list when no
model is currently loaded.

Run the real conversation pipeline integration test:

```powershell
uv run pytest tests/test_conversation_service.py -v -s
```

Expected result: Friday sends the prompt through the real
`ConversationService -> OllamaClient -> Ollama -> model` path, persists the user
and assistant messages in PostgreSQL, cleans up the test conversation, and passes.

## Verified conversation milestone

The real conversation pipeline has produced and verified:

```text
===== FRIDAY ANSWER =====
Conversation pipeline working
PASSED
```

## Verified Windows GPU baseline

The known-good local configuration is:

```text
Ollama: 0.24.0
Model: qwen2.5:3b
GPU: NVIDIA GTX 950M 4 GB
Inference: 100% GPU verified with ollama ps
Context: 4096
```

Ollama 0.32.14 failed during GPU discovery on this machine with Windows error
`0xc0000005`. Forcing CUDA 12 did not fix it, and Vulkan did not fix it.
Downgrading to Ollama 0.24.0 restored GPU acceleration. Do not upgrade Ollama
without verifying GPU acceleration afterward.

Qwen3 was removed because its thinking behavior was undesirable for Friday. The
current `qwen2.5:3b` path stores user messages and final assistant responses only;
thinking or reasoning traces are neither displayed nor persisted.

Verify the approved runtime and GPU offload with:

```powershell
ollama --version
ollama run qwen2.5:3b "Reply only with: Hello"
ollama ps
```

Expected results are Ollama `0.24.0`, a final response of `Hello`, and
`PROCESSOR = 100% GPU` in `ollama ps`.
