import re
from dataclasses import dataclass
from enum import Enum

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_IDENTIFIER_MAX_LENGTH = 64
_DESCRIPTION_MAX_LENGTH = 500
_PARAMETER_DESCRIPTION_MAX_LENGTH = 300
_ALLOWED_VALUES_MAX_LENGTH = 50


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    MODIFY = "modify"
    DESTRUCTIVE = "destructive"


class ToolParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class ToolParameter:
    name: str
    description: str
    parameter_type: ToolParameterType
    required: bool = True
    allowed_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_name = _normalize_identifier(self.name, field_name="name")
        normalized_description = _normalize_text(
            self.description,
            field_name="description",
            maximum_length=_PARAMETER_DESCRIPTION_MAX_LENGTH,
        )

        if not isinstance(self.parameter_type, ToolParameterType):
            raise ValueError(  # noqa: TRY004
                "parameter_type must be a ToolParameterType value"
            )

        if not isinstance(self.required, bool):
            raise ValueError("required must be a bool")  # noqa: TRY004

        normalized_values = _normalize_allowed_values(self.allowed_values)
        if (
            normalized_values
            and self.parameter_type is not ToolParameterType.STRING
        ):
            raise ValueError(
                "allowed_values are supported only for string parameters"
            )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)
        object.__setattr__(self, "allowed_values", normalized_values)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    category: str
    risk: ToolRisk
    parameters: tuple[ToolParameter, ...] = ()

    def __post_init__(self) -> None:
        normalized_name = _normalize_identifier(self.name, field_name="name")
        normalized_description = _normalize_description(self.description)
        normalized_category = _normalize_identifier(
            self.category,
            field_name="category",
        )

        if not isinstance(self.risk, ToolRisk):
            raise ValueError("risk must be a ToolRisk value")  # noqa: TRY004

        if not isinstance(self.parameters, tuple):
            raise ValueError("parameters must be a tuple")  # noqa: TRY004

        parameter_names: set[str] = set()
        for parameter in self.parameters:
            if not isinstance(parameter, ToolParameter):
                raise ValueError(  # noqa: TRY004
                    "parameters must contain only ToolParameter values"
                )
            if parameter.name in parameter_names:
                raise ValueError(
                    f"duplicate tool parameter name: '{parameter.name}'"
                )
            parameter_names.add(parameter.name)

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)
        object.__setattr__(self, "category", normalized_category)


def _normalize_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")  # noqa: TRY004

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    if len(normalized) > _IDENTIFIER_MAX_LENGTH:
        raise ValueError(
            f"{field_name} must be at most {_IDENTIFIER_MAX_LENGTH} characters"
        )

    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must be lowercase snake_case and start with a letter"
        )

    return normalized


def _normalize_description(value: str) -> str:
    return _normalize_text(
        value,
        field_name="description",
        maximum_length=_DESCRIPTION_MAX_LENGTH,
    )


def _normalize_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")  # noqa: TRY004

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    if len(normalized) > maximum_length:
        raise ValueError(
            f"{field_name} must be at most {maximum_length} characters"
        )

    return normalized


def _normalize_allowed_values(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError("allowed_values must be a tuple")  # noqa: TRY004

    if len(values) > _ALLOWED_VALUES_MAX_LENGTH:
        raise ValueError(
            f"allowed_values must contain at most {_ALLOWED_VALUES_MAX_LENGTH} entries"
        )

    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(  # noqa: TRY004
                "allowed_values entries must be strings"
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("allowed_values entries must not be empty")
        if normalized in seen_values:
            raise ValueError(f"duplicate allowed value: '{normalized}'")

        normalized_values.append(normalized)
        seen_values.add(normalized)

    return tuple(normalized_values)
