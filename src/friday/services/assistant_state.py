from enum import Enum


class AssistantState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    STREAMING = "streaming"
    SPEAKING = "speaking"
    TOOL_RUNNING = "tool_running"
    OFFLINE = "offline"
    ERROR = "error"


class InvalidStateTransition(ValueError):
    """Raised when an assistant state transition is not allowed."""


_ALLOWED_TRANSITIONS: dict[AssistantState, frozenset[AssistantState]] = {
    AssistantState.IDLE: frozenset(
        {
            AssistantState.LISTENING,
            AssistantState.PROCESSING,
            AssistantState.OFFLINE,
            AssistantState.ERROR,
        }
    ),
    AssistantState.LISTENING: frozenset(
        {
            AssistantState.IDLE,
            AssistantState.PROCESSING,
            AssistantState.OFFLINE,
            AssistantState.ERROR,
        }
    ),
    AssistantState.PROCESSING: frozenset(
        {
            AssistantState.IDLE,
            AssistantState.STREAMING,
            AssistantState.SPEAKING,
            AssistantState.TOOL_RUNNING,
            AssistantState.OFFLINE,
            AssistantState.ERROR,
        }
    ),
    AssistantState.STREAMING: frozenset(
        {
            AssistantState.IDLE,
            AssistantState.SPEAKING,
            AssistantState.TOOL_RUNNING,
            AssistantState.OFFLINE,
            AssistantState.ERROR,
        }
    ),
    AssistantState.SPEAKING: frozenset(
        {
            AssistantState.IDLE,
            AssistantState.LISTENING,
            AssistantState.OFFLINE,
            AssistantState.ERROR,
        }
    ),
    AssistantState.TOOL_RUNNING: frozenset(
        {
            AssistantState.IDLE,
            AssistantState.PROCESSING,
            AssistantState.STREAMING,
            AssistantState.SPEAKING,
            AssistantState.OFFLINE,
            AssistantState.ERROR,
        }
    ),
    AssistantState.OFFLINE: frozenset(
        {
            AssistantState.IDLE,
            AssistantState.ERROR,
        }
    ),
    AssistantState.ERROR: frozenset(
        {
            AssistantState.IDLE,
            AssistantState.OFFLINE,
        }
    ),
}


class AssistantStateMachine:
    """Maintain Friday's current runtime state in memory."""

    def __init__(self) -> None:
        self._state = AssistantState.IDLE

    def get(self) -> AssistantState:
        return self._state

    def can_transition(self, target: AssistantState) -> bool:
        self._validate_target(target)
        return target is self._state or target in _ALLOWED_TRANSITIONS[self._state]

    def transition(self, target: AssistantState) -> AssistantState:
        self._validate_target(target)

        if not self.can_transition(target):
            raise InvalidStateTransition(
                "Invalid assistant state transition: "
                f"{self._state.value} -> {target.value}"
            )

        self._state = target
        return self._state

    def reset(self) -> AssistantState:
        self._state = AssistantState.IDLE
        return self._state

    @staticmethod
    def _validate_target(target: AssistantState) -> None:
        if not isinstance(target, AssistantState):
            raise TypeError("Assistant state target must be an AssistantState.")
