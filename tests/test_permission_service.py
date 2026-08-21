from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from friday.services.permission_service import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionRequiredError,
    PermissionResult,
    PermissionService,
    PermissionServiceError,
)
from friday.tools import ToolDefinition, ToolRisk


def make_tool(risk: ToolRisk) -> ToolDefinition:
    return ToolDefinition(
        name=f"{risk.value}_tool",
        description="A permission policy test tool.",
        category="tests",
        risk=risk,
    )


def test_permission_decision_values_are_stable() -> None:
    assert PermissionDecision.ALLOW.value == "allow"
    assert PermissionDecision.REQUIRE_APPROVAL.value == "require_approval"
    assert PermissionDecision.DENY.value == "deny"


def test_read_only_is_allowed_without_approval() -> None:
    result = PermissionService().evaluate(make_tool(ToolRisk.READ_ONLY))

    assert result.decision is PermissionDecision.ALLOW
    assert "read_only_tool" in result.reason


def test_modify_requires_approval_without_user_approval() -> None:
    result = PermissionService().evaluate(make_tool(ToolRisk.MODIFY))

    assert result.decision is PermissionDecision.REQUIRE_APPROVAL
    assert result.reason == "Tool 'modify_tool' requires explicit approval."


def test_modify_is_allowed_with_explicit_approval() -> None:
    result = PermissionService().evaluate(
        make_tool(ToolRisk.MODIFY),
        user_approved=True,
    )

    assert result.decision is PermissionDecision.ALLOW


@pytest.mark.parametrize("user_approved", [False, True])
def test_destructive_is_denied_even_with_approval(
    user_approved: bool,
) -> None:
    result = PermissionService().evaluate(
        make_tool(ToolRisk.DESTRUCTIVE),
        user_approved=user_approved,
    )

    assert result.decision is PermissionDecision.DENY
    assert result.reason == (
        "Tool 'destructive_tool' is denied by the current permission policy."
    )


def test_authorize_returns_none_for_allowed_tool() -> None:
    result = PermissionService().authorize(make_tool(ToolRisk.READ_ONLY))

    assert result is None


def test_authorize_raises_permission_required_error() -> None:
    with pytest.raises(
        PermissionRequiredError,
        match="Tool 'modify_tool' requires explicit approval",
    ) as exc_info:
        PermissionService().authorize(make_tool(ToolRisk.MODIFY))

    assert isinstance(exc_info.value, PermissionServiceError)


@pytest.mark.parametrize("user_approved", [False, True])
def test_authorize_raises_permission_denied_error(
    user_approved: bool,
) -> None:
    with pytest.raises(
        PermissionDeniedError,
        match="Tool 'destructive_tool' is denied",
    ) as exc_info:
        PermissionService().authorize(
            make_tool(ToolRisk.DESTRUCTIVE),
            user_approved=user_approved,
        )

    assert isinstance(exc_info.value, PermissionServiceError)


def test_permission_result_normalizes_reason_and_is_immutable() -> None:
    result = PermissionResult(
        decision=PermissionDecision.ALLOW,
        reason="  Explicit reason.  ",
    )

    assert result.reason == "Explicit reason."

    with pytest.raises(FrozenInstanceError):
        result.reason = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize("decision", ["allow", None, 1])
def test_permission_result_rejects_invalid_decision(decision: Any) -> None:
    with pytest.raises(ValueError, match="decision"):
        PermissionResult(decision=decision, reason="A reason.")


@pytest.mark.parametrize("reason", ["", "   ", None, 1])
def test_permission_result_rejects_invalid_reason(reason: Any) -> None:
    with pytest.raises(ValueError, match="reason"):
        PermissionResult(decision=PermissionDecision.ALLOW, reason=reason)


@pytest.mark.parametrize("method_name", ["evaluate", "authorize"])
def test_permission_service_rejects_invalid_tool(method_name: str) -> None:
    method = getattr(PermissionService(), method_name)

    with pytest.raises(ValueError, match="ToolDefinition"):
        method(object())


@pytest.mark.parametrize("method_name", ["evaluate", "authorize"])
@pytest.mark.parametrize("user_approved", [1, 0, "yes", None])
def test_permission_service_rejects_invalid_approval_values(
    method_name: str,
    user_approved: Any,
) -> None:
    method = getattr(PermissionService(), method_name)

    with pytest.raises(ValueError, match="user_approved"):
        method(
            make_tool(ToolRisk.MODIFY),
            user_approved=user_approved,
        )


def test_approval_is_not_stored_between_calls() -> None:
    service = PermissionService()
    tool = make_tool(ToolRisk.MODIFY)

    approved = service.evaluate(tool, user_approved=True)
    later = service.evaluate(tool, user_approved=False)

    assert approved.decision is PermissionDecision.ALLOW
    assert later.decision is PermissionDecision.REQUIRE_APPROVAL
