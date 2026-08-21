from friday.tools.models import ToolDefinition, ToolRisk
from friday.tools.registry import ToolRegistry


def tool_to_ollama_schema(tool: ToolDefinition) -> dict[str, object]:
    if not isinstance(tool, ToolDefinition):
        raise ValueError("tool must be a ToolDefinition")  # noqa: TRY004

    properties: dict[str, object] = {}
    required: list[str] = []

    for parameter in tool.parameters:
        property_schema: dict[str, object] = {
            "type": parameter.parameter_type.value,
            "description": parameter.description,
        }
        if parameter.allowed_values:
            property_schema["enum"] = list(parameter.allowed_values)

        properties[parameter.name] = property_schema
        if parameter.required:
            required.append(parameter.name)

    parameters_schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema,
        },
    }


def registry_to_ollama_tools(
    registry: ToolRegistry,
    *,
    allowed_risks: frozenset[ToolRisk],
) -> list[dict[str, object]]:
    if not isinstance(registry, ToolRegistry):
        raise ValueError("registry must be a ToolRegistry")  # noqa: TRY004
    if not isinstance(allowed_risks, frozenset) or any(
        not isinstance(risk, ToolRisk) for risk in allowed_risks
    ):
        raise ValueError(
            "allowed_risks must be a frozenset of ToolRisk values"
        )

    return [
        tool_to_ollama_schema(tool)
        for tool in registry.list_tools()
        if tool.risk in allowed_risks
    ]
