from friday.tools.models import ToolDefinition


class ToolRegistryError(Exception):
    """Base exception for tool registry operations."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a tool name is already registered."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a required tool is not registered."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not isinstance(tool, ToolDefinition):
            raise ValueError("tool must be a ToolDefinition")  # noqa: TRY004

        if tool.name in self._tools:
            raise DuplicateToolError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        normalized_name = _normalize_lookup_name(name)
        return self._tools.get(normalized_name)

    def require(self, name: str) -> ToolDefinition:
        normalized_name = _normalize_lookup_name(name)
        tool = self._tools.get(normalized_name)

        if tool is None:
            raise ToolNotFoundError(
                f"Tool '{normalized_name}' is not registered."
            )

        return tool

    def unregister(self, name: str) -> bool:
        normalized_name = _normalize_lookup_name(name)
        return self._tools.pop(normalized_name, None) is not None

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            sorted(
                self._tools.values(),
                key=lambda tool: tool.name,
            )
        )

    def __len__(self) -> int:
        return len(self._tools)


def _normalize_lookup_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("tool name must be a string")  # noqa: TRY004

    normalized = name.strip()

    if not normalized:
        raise ValueError("tool name must not be empty")

    return normalized
