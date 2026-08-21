import re
from dataclasses import dataclass
from enum import Enum

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_IDENTIFIER_MAX_LENGTH = 64
_DESCRIPTION_MAX_LENGTH = 500


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    MODIFY = "modify"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    category: str
    risk: ToolRisk

    def __post_init__(self) -> None:
        normalized_name = _normalize_identifier(self.name, field_name="name")
        normalized_description = _normalize_description(self.description)
        normalized_category = _normalize_identifier(
            self.category,
            field_name="category",
        )

        if not isinstance(self.risk, ToolRisk):
            raise ValueError("risk must be a ToolRisk value")  # noqa: TRY004

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
    if not isinstance(value, str):
        raise ValueError("description must be a string")  # noqa: TRY004

    normalized = value.strip()

    if not normalized:
        raise ValueError("description must not be empty")

    if len(normalized) > _DESCRIPTION_MAX_LENGTH:
        raise ValueError(
            f"description must be at most {_DESCRIPTION_MAX_LENGTH} characters"
        )

    return normalized
