from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from friday.tools import (
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolRisk,
    registry_to_ollama_tools,
    tool_to_ollama_schema,
)


def make_parameter(**overrides: Any) -> ToolParameter:
    values: dict[str, Any] = {
        "name": "app_name",
        "description": "Canonical application name.",
        "parameter_type": ToolParameterType.STRING,
    }
    values.update(overrides)
    return ToolParameter(**values)


def make_tool(
    name: str = "open_app",
    *,
    risk: ToolRisk = ToolRisk.MODIFY,
    parameters: tuple[ToolParameter, ...] = (),
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Description for {name}.",
        category="tests",
        risk=risk,
        parameters=parameters,
    )


def test_backward_compatible_tool_definition_has_no_parameters() -> None:
    tool = ToolDefinition(
        name="system_info",
        description="Read system information.",
        category="system",
        risk=ToolRisk.READ_ONLY,
    )

    assert tool.parameters == ()


def test_no_parameter_schema_omits_required() -> None:
    schema = tool_to_ollama_schema(
        make_tool("get_system_info", risk=ToolRisk.READ_ONLY)
    )

    assert schema == {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Description for get_system_info.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


def test_required_string_parameter_schema() -> None:
    parameter = make_parameter()

    schema = tool_to_ollama_schema(make_tool(parameters=(parameter,)))

    assert schema["function"] == {
        "name": "open_app",
        "description": "Description for open_app.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Canonical application name.",
                }
            },
            "required": ["app_name"],
        },
    }


@pytest.mark.parametrize(
    ("parameter_type", "expected_type"),
    [
        (ToolParameterType.STRING, "string"),
        (ToolParameterType.INTEGER, "integer"),
        (ToolParameterType.NUMBER, "number"),
        (ToolParameterType.BOOLEAN, "boolean"),
    ],
)
def test_parameter_types_convert_to_json_schema(
    parameter_type: ToolParameterType,
    expected_type: str,
) -> None:
    parameter = make_parameter(parameter_type=parameter_type)
    schema = tool_to_ollama_schema(make_tool(parameters=(parameter,)))
    function = schema["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)

    assert properties["app_name"] == {
        "type": expected_type,
        "description": "Canonical application name.",
    }


def test_optional_parameter_is_not_listed_as_required() -> None:
    parameter = make_parameter(required=False)
    schema = tool_to_ollama_schema(make_tool(parameters=(parameter,)))
    function = schema["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)

    assert "required" not in parameters


def test_allowed_values_are_normalized_and_converted_to_enum() -> None:
    parameter = make_parameter(
        allowed_values=("  Notepad  ", "calculator"),
    )
    schema = tool_to_ollama_schema(make_tool(parameters=(parameter,)))

    assert parameter.allowed_values == ("Notepad", "calculator")
    function = schema["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["app_name"] == {
        "type": "string",
        "description": "Canonical application name.",
        "enum": ["Notepad", "calculator"],
    }


def test_schema_conversion_returns_new_deterministic_dictionaries() -> None:
    tool = make_tool(parameters=(make_parameter(),))

    first = tool_to_ollama_schema(tool)
    second = tool_to_ollama_schema(tool)

    assert first == second
    assert first is not second
    assert first["function"] is not second["function"]


def test_registry_conversion_filters_risks_and_sorts_names() -> None:
    registry = ToolRegistry()
    registry.register(make_tool("z_read", risk=ToolRisk.READ_ONLY))
    registry.register(make_tool("modify_tool", risk=ToolRisk.MODIFY))
    registry.register(make_tool("a_read", risk=ToolRisk.READ_ONLY))
    registry.register(make_tool("destroy_tool", risk=ToolRisk.DESTRUCTIVE))

    schemas = registry_to_ollama_tools(
        registry,
        allowed_risks=frozenset({ToolRisk.READ_ONLY}),
    )

    assert [schema["function"]["name"] for schema in schemas] == [  # type: ignore[index]
        "a_read",
        "z_read",
    ]


@pytest.mark.parametrize(
    "name",
    ["", "   ", "AppName", "app-name", "2app", "app.name", "a" * 65, 1],
)
def test_parameter_rejects_invalid_name(name: Any) -> None:
    with pytest.raises(ValueError, match="name"):
        make_parameter(name=name)


@pytest.mark.parametrize("description", ["", "   ", "x" * 301, None, 1])
def test_parameter_rejects_invalid_description(description: Any) -> None:
    with pytest.raises(ValueError, match="description"):
        make_parameter(description=description)


@pytest.mark.parametrize("parameter_type", ["string", None, 1])
def test_parameter_rejects_invalid_type(parameter_type: Any) -> None:
    with pytest.raises(ValueError, match="parameter_type"):
        make_parameter(parameter_type=parameter_type)


@pytest.mark.parametrize("required", [0, 1, "yes", None])
def test_parameter_rejects_invalid_required(required: Any) -> None:
    with pytest.raises(ValueError, match="required"):
        make_parameter(required=required)


@pytest.mark.parametrize("allowed_values", [[], "notepad", None, {"notepad"}])
def test_parameter_requires_allowed_values_tuple(allowed_values: Any) -> None:
    with pytest.raises(ValueError, match="allowed_values"):
        make_parameter(allowed_values=allowed_values)


@pytest.mark.parametrize(
    "allowed_values",
    [("",), ("   ",), ("notepad", 1), ("notepad", " notepad ")],
)
def test_parameter_rejects_invalid_allowed_value_entries(
    allowed_values: Any,
) -> None:
    with pytest.raises(ValueError, match="allowed|duplicate"):
        make_parameter(allowed_values=allowed_values)


def test_parameter_limits_allowed_value_count() -> None:
    with pytest.raises(ValueError, match="at most 50"):
        make_parameter(allowed_values=tuple(f"value_{index}" for index in range(51)))


@pytest.mark.parametrize(
    "parameter_type",
    [
        ToolParameterType.INTEGER,
        ToolParameterType.NUMBER,
        ToolParameterType.BOOLEAN,
    ],
)
def test_allowed_values_are_string_only(
    parameter_type: ToolParameterType,
) -> None:
    with pytest.raises(ValueError, match="only for string"):
        make_parameter(
            parameter_type=parameter_type,
            allowed_values=("value",),
        )


def test_parameter_is_immutable() -> None:
    parameter = make_parameter()

    with pytest.raises(FrozenInstanceError):
        parameter.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("parameters", [[], None, {"parameter"}])
def test_tool_definition_requires_parameter_tuple(parameters: Any) -> None:
    with pytest.raises(ValueError, match="parameters must be a tuple"):
        make_tool(parameters=parameters)


def test_tool_definition_requires_tool_parameter_members() -> None:
    with pytest.raises(ValueError, match="ToolParameter"):
        make_tool(parameters=(object(),))  # type: ignore[arg-type]


def test_tool_definition_rejects_duplicate_parameter_names() -> None:
    with pytest.raises(ValueError, match="duplicate tool parameter"):
        make_tool(
            parameters=(
                make_parameter(name="app_name"),
                make_parameter(name=" app_name "),
            )
        )
