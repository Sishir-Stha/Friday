from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from friday.tools import (
    DuplicateToolError,
    ToolDefinition,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
    ToolRisk,
)


def make_tool(
    name: str = "system_info",
    *,
    description: str = "Read basic system information.",
    category: str = "system",
    risk: ToolRisk = ToolRisk.READ_ONLY,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        category=category,
        risk=risk,
    )


def test_tool_risk_values_are_stable() -> None:
    assert ToolRisk.READ_ONLY.value == "read_only"
    assert ToolRisk.MODIFY.value == "modify"
    assert ToolRisk.DESTRUCTIVE.value == "destructive"


def test_valid_tool_definition() -> None:
    tool = make_tool()

    assert tool.name == "system_info"
    assert tool.description == "Read basic system information."
    assert tool.category == "system"
    assert tool.risk is ToolRisk.READ_ONLY


def test_tool_definition_normalizes_surrounding_whitespace() -> None:
    tool = ToolDefinition(
        name="  system_info  ",
        description="  Read basic system information.  ",
        category="  system  ",
        risk=ToolRisk.READ_ONLY,
    )

    assert tool.name == "system_info"
    assert tool.description == "Read basic system information."
    assert tool.category == "system"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "OpenApp",
        "open-app",
        "2tool",
        "tool name",
        "tool.name",
        "a" * 65,
        123,
        None,
    ],
)
def test_invalid_tool_names_raise_value_error(name: Any) -> None:
    with pytest.raises(ValueError, match="name"):
        make_tool(name=name)


@pytest.mark.parametrize(
    "name",
    ["system_info", "get_cpu_usage", "list_files", "open_app", "file2"],
)
def test_valid_tool_name_formats(name: str) -> None:
    assert make_tool(name=name).name == name


def test_tool_name_accepts_maximum_length() -> None:
    name = "a" + "1" * 63

    assert make_tool(name=name).name == name


@pytest.mark.parametrize("description", ["", "   ", "x" * 501, 123, None])
def test_invalid_descriptions_raise_value_error(description: Any) -> None:
    with pytest.raises(ValueError, match="description"):
        make_tool(description=description)


def test_description_accepts_maximum_length() -> None:
    description = "x" * 500

    assert make_tool(description=description).description == description


@pytest.mark.parametrize(
    "category",
    [
        "",
        "   ",
        "System",
        "system-info",
        "2system",
        "system info",
        "system.info",
        "a" * 65,
        123,
        None,
    ],
)
def test_invalid_categories_raise_value_error(category: Any) -> None:
    with pytest.raises(ValueError, match="category"):
        make_tool(category=category)


def test_category_accepts_maximum_length() -> None:
    category = "a" + "1" * 63

    assert make_tool(category=category).category == category


@pytest.mark.parametrize("risk", ["read_only", "modify", "destructive", None])
def test_invalid_risk_requires_tool_risk_instance(risk: Any) -> None:
    with pytest.raises(ValueError, match="risk"):
        make_tool(risk=risk)


def test_tool_definition_is_immutable() -> None:
    tool = make_tool()

    with pytest.raises(FrozenInstanceError):
        tool.name = "different_tool"  # type: ignore[misc]


def test_register_and_get_returns_exact_definition() -> None:
    registry = ToolRegistry()
    tool = make_tool()

    registry.register(tool)

    assert registry.get("system_info") is tool
    assert registry.get("  system_info  ") is tool


def test_duplicate_registration_of_same_object_is_rejected() -> None:
    registry = ToolRegistry()
    tool = make_tool()
    registry.register(tool)

    with pytest.raises(
        DuplicateToolError,
        match="Tool 'system_info' is already registered",
    ) as exc_info:
        registry.register(tool)

    assert isinstance(exc_info.value, ToolRegistryError)
    assert registry.get("system_info") is tool


def test_duplicate_registration_of_distinct_definition_is_rejected() -> None:
    registry = ToolRegistry()
    original = make_tool()
    duplicate = make_tool(description="A different description.")
    registry.register(original)

    with pytest.raises(DuplicateToolError):
        registry.register(duplicate)

    assert registry.get("system_info") is original


def test_unknown_get_returns_none() -> None:
    assert ToolRegistry().get("missing_tool") is None


def test_require_existing_returns_exact_definition() -> None:
    registry = ToolRegistry()
    tool = make_tool()
    registry.register(tool)

    assert registry.require("  system_info  ") is tool


def test_require_missing_raises_tool_not_found_error() -> None:
    with pytest.raises(
        ToolNotFoundError,
        match="Tool 'missing_tool' is not registered",
    ) as exc_info:
        ToolRegistry().require("missing_tool")

    assert isinstance(exc_info.value, ToolRegistryError)


def test_unregister_returns_true_then_false() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())

    assert registry.unregister("  system_info  ") is True
    assert registry.unregister("system_info") is False
    assert registry.get("system_info") is None


def test_list_tools_returns_alphabetical_immutable_tuple() -> None:
    registry = ToolRegistry()
    registry.register(make_tool("z_tool"))
    registry.register(make_tool("a_tool"))
    registry.register(make_tool("m_tool"))

    tools = registry.list_tools()

    assert isinstance(tools, tuple)
    assert [tool.name for tool in tools] == ["a_tool", "m_tool", "z_tool"]


def test_registry_instances_are_independent() -> None:
    registry_a = ToolRegistry()
    registry_b = ToolRegistry()
    registry_a.register(make_tool())

    assert len(registry_a) == 1
    assert len(registry_b) == 0
    assert registry_b.get("system_info") is None


def test_registry_length_tracks_registration_lifecycle() -> None:
    registry = ToolRegistry()
    assert len(registry) == 0

    registry.register(make_tool("system_info"))
    registry.register(make_tool("list_files", category="files"))
    assert len(registry) == 2

    assert registry.unregister("system_info") is True
    assert len(registry) == 1


@pytest.mark.parametrize("invalid_tool", [None, "system_info", object()])
def test_register_rejects_non_definition_objects(invalid_tool: Any) -> None:
    with pytest.raises(ValueError, match="ToolDefinition"):
        ToolRegistry().register(invalid_tool)


@pytest.mark.parametrize("method_name", ["get", "require", "unregister"])
@pytest.mark.parametrize("invalid_name", [None, 123, object()])
def test_lookup_operations_reject_non_string_names(
    method_name: str,
    invalid_name: Any,
) -> None:
    method = getattr(ToolRegistry(), method_name)

    with pytest.raises(ValueError, match="tool name must be a string"):
        method(invalid_name)


@pytest.mark.parametrize("method_name", ["get", "require", "unregister"])
@pytest.mark.parametrize("empty_name", ["", "   "])
def test_lookup_operations_reject_empty_names(
    method_name: str,
    empty_name: str,
) -> None:
    method = getattr(ToolRegistry(), method_name)

    with pytest.raises(ValueError, match="tool name must not be empty"):
        method(empty_name)


def test_registry_exposes_no_execution_api() -> None:
    registry = ToolRegistry()

    for method_name in ("execute", "run", "invoke", "call_tool", "execute_tool"):
        assert not hasattr(registry, method_name)
