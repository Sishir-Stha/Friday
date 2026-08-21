from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    active_module: str | None = None
    active_window: str | None = None
    selected_item: str | None = None
    current_conversation_id: int | None = None
    recent_tool_result: str | None = None


class _UnsetType:
    __slots__ = ()


_UNSET = _UnsetType()


class ContextManager:
    """Maintain Friday's ephemeral runtime context in memory."""

    def __init__(self) -> None:
        self._snapshot = ContextSnapshot()

    def get(self) -> ContextSnapshot:
        return self._snapshot

    def update(
        self,
        *,
        active_module: str | None | _UnsetType = _UNSET,
        active_window: str | None | _UnsetType = _UNSET,
        selected_item: str | None | _UnsetType = _UNSET,
        current_conversation_id: int | None | _UnsetType = _UNSET,
        recent_tool_result: str | None | _UnsetType = _UNSET,
    ) -> ContextSnapshot:
        current = self._snapshot

        next_snapshot = ContextSnapshot(
            active_module=self._normalize_string(
                active_module,
                current.active_module,
            ),
            active_window=self._normalize_string(
                active_window,
                current.active_window,
            ),
            selected_item=self._normalize_string(
                selected_item,
                current.selected_item,
            ),
            current_conversation_id=self._normalize_conversation_id(
                current_conversation_id,
                current.current_conversation_id,
            ),
            recent_tool_result=self._normalize_string(
                recent_tool_result,
                current.recent_tool_result,
            ),
        )

        self._snapshot = next_snapshot
        return next_snapshot

    def clear(self) -> ContextSnapshot:
        self._snapshot = ContextSnapshot()
        return self._snapshot

    @staticmethod
    def _normalize_string(
        value: str | None | _UnsetType,
        current: str | None,
    ) -> str | None:
        if isinstance(value, _UnsetType):
            return current

        if value is None:
            return None

        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _normalize_conversation_id(
        value: int | None | _UnsetType,
        current: int | None,
    ) -> int | None:
        if isinstance(value, _UnsetType):
            return current

        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Conversation ID must be a positive integer.")

        return value
