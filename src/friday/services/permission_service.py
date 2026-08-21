from dataclasses import dataclass
from enum import Enum

from friday.tools.models import ToolDefinition, ToolRisk


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PermissionResult:
    decision: PermissionDecision
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PermissionDecision):
            raise ValueError(  # noqa: TRY004
                "decision must be a PermissionDecision value"
            )

        if not isinstance(self.reason, str):
            raise ValueError("reason must be a string")  # noqa: TRY004

        normalized_reason = self.reason.strip()
        if not normalized_reason:
            raise ValueError("reason must not be empty")

        object.__setattr__(self, "reason", normalized_reason)


class PermissionServiceError(Exception):
    """Base exception for tool permission failures."""


class PermissionRequiredError(PermissionServiceError):
    """Raised when a tool requires explicit per-call approval."""


class PermissionDeniedError(PermissionServiceError):
    """Raised when the current policy denies a tool."""


class PermissionService:
    def evaluate(
        self,
        tool: ToolDefinition,
        *,
        user_approved: bool = False,
    ) -> PermissionResult:
        self._validate_inputs(tool, user_approved=user_approved)

        if tool.risk is ToolRisk.READ_ONLY:
            return PermissionResult(
                decision=PermissionDecision.ALLOW,
                reason=f"Tool '{tool.name}' is read-only and allowed.",
            )

        if tool.risk is ToolRisk.MODIFY:
            if user_approved:
                return PermissionResult(
                    decision=PermissionDecision.ALLOW,
                    reason=f"Tool '{tool.name}' has explicit approval.",
                )

            return PermissionResult(
                decision=PermissionDecision.REQUIRE_APPROVAL,
                reason=f"Tool '{tool.name}' requires explicit approval.",
            )

        return PermissionResult(
            decision=PermissionDecision.DENY,
            reason=(
                f"Tool '{tool.name}' is denied by the current permission policy."
            ),
        )

    def authorize(
        self,
        tool: ToolDefinition,
        *,
        user_approved: bool = False,
    ) -> None:
        result = self.evaluate(tool, user_approved=user_approved)

        if result.decision is PermissionDecision.ALLOW:
            return

        if result.decision is PermissionDecision.REQUIRE_APPROVAL:
            raise PermissionRequiredError(result.reason)

        raise PermissionDeniedError(result.reason)

    @staticmethod
    def _validate_inputs(
        tool: ToolDefinition,
        *,
        user_approved: bool,
    ) -> None:
        if not isinstance(tool, ToolDefinition):
            raise ValueError("tool must be a ToolDefinition")  # noqa: TRY004

        if not isinstance(user_approved, bool):
            raise ValueError("user_approved must be a bool")  # noqa: TRY004
