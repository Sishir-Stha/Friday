from collections.abc import Callable

import pytest

from friday.services.assistant_state import (
    AssistantState,
    AssistantStateMachine,
    InvalidStateTransition,
)


def test_initial_state_is_idle() -> None:
    machine = AssistantStateMachine()

    assert machine.get() is AssistantState.IDLE


def test_normal_conversation_transitions() -> None:
    machine = AssistantStateMachine()

    assert machine.transition(AssistantState.PROCESSING) is AssistantState.PROCESSING
    assert machine.transition(AssistantState.STREAMING) is AssistantState.STREAMING
    assert machine.transition(AssistantState.IDLE) is AssistantState.IDLE


def test_future_voice_transitions() -> None:
    machine = AssistantStateMachine()

    assert machine.transition(AssistantState.LISTENING) is AssistantState.LISTENING
    assert machine.transition(AssistantState.PROCESSING) is AssistantState.PROCESSING
    assert machine.transition(AssistantState.SPEAKING) is AssistantState.SPEAKING
    assert machine.transition(AssistantState.IDLE) is AssistantState.IDLE


def test_tool_transitions() -> None:
    machine = AssistantStateMachine()

    expected_states = [
        AssistantState.PROCESSING,
        AssistantState.TOOL_RUNNING,
        AssistantState.PROCESSING,
        AssistantState.STREAMING,
        AssistantState.IDLE,
    ]

    assert [machine.transition(state) for state in expected_states] == expected_states


def test_offline_transition_and_recovery() -> None:
    machine = AssistantStateMachine()

    assert machine.transition(AssistantState.OFFLINE) is AssistantState.OFFLINE
    assert machine.transition(AssistantState.IDLE) is AssistantState.IDLE


def test_error_transition_and_recovery() -> None:
    machine = AssistantStateMachine()
    machine.transition(AssistantState.PROCESSING)

    assert machine.transition(AssistantState.ERROR) is AssistantState.ERROR
    assert machine.transition(AssistantState.IDLE) is AssistantState.IDLE


def test_invalid_transition_does_not_change_state() -> None:
    machine = AssistantStateMachine()

    with pytest.raises(
        InvalidStateTransition,
        match="Invalid assistant state transition: idle -> speaking",
    ):
        machine.transition(AssistantState.SPEAKING)

    assert machine.get() is AssistantState.IDLE


def test_same_state_transition_is_idempotent() -> None:
    machine = AssistantStateMachine()

    assert machine.can_transition(AssistantState.IDLE) is True
    assert machine.transition(AssistantState.IDLE) is AssistantState.IDLE
    assert machine.get() is AssistantState.IDLE


def test_can_transition_reports_valid_and_invalid_targets() -> None:
    machine = AssistantStateMachine()

    assert machine.can_transition(AssistantState.PROCESSING) is True
    assert machine.can_transition(AssistantState.SPEAKING) is False


@pytest.mark.parametrize(
    ("source_path", "source", "allowed"),
    [
        (
            [],
            AssistantState.IDLE,
            {
                AssistantState.LISTENING,
                AssistantState.PROCESSING,
                AssistantState.TOOL_RUNNING,
                AssistantState.OFFLINE,
                AssistantState.ERROR,
            },
        ),
        (
            [AssistantState.LISTENING],
            AssistantState.LISTENING,
            {
                AssistantState.IDLE,
                AssistantState.PROCESSING,
                AssistantState.OFFLINE,
                AssistantState.ERROR,
            },
        ),
        (
            [AssistantState.PROCESSING],
            AssistantState.PROCESSING,
            {
                AssistantState.IDLE,
                AssistantState.STREAMING,
                AssistantState.SPEAKING,
                AssistantState.TOOL_RUNNING,
                AssistantState.OFFLINE,
                AssistantState.ERROR,
            },
        ),
        (
            [AssistantState.PROCESSING, AssistantState.STREAMING],
            AssistantState.STREAMING,
            {
                AssistantState.IDLE,
                AssistantState.SPEAKING,
                AssistantState.TOOL_RUNNING,
                AssistantState.OFFLINE,
                AssistantState.ERROR,
            },
        ),
        (
            [AssistantState.PROCESSING, AssistantState.SPEAKING],
            AssistantState.SPEAKING,
            {
                AssistantState.IDLE,
                AssistantState.LISTENING,
                AssistantState.OFFLINE,
                AssistantState.ERROR,
            },
        ),
        (
            [AssistantState.PROCESSING, AssistantState.TOOL_RUNNING],
            AssistantState.TOOL_RUNNING,
            {
                AssistantState.IDLE,
                AssistantState.PROCESSING,
                AssistantState.STREAMING,
                AssistantState.SPEAKING,
                AssistantState.OFFLINE,
                AssistantState.ERROR,
            },
        ),
        (
            [AssistantState.OFFLINE],
            AssistantState.OFFLINE,
            {
                AssistantState.IDLE,
                AssistantState.ERROR,
            },
        ),
        (
            [AssistantState.ERROR],
            AssistantState.ERROR,
            {
                AssistantState.IDLE,
                AssistantState.OFFLINE,
            },
        ),
    ],
)
def test_transition_map_matches_rules(
    source_path: list[AssistantState],
    source: AssistantState,
    allowed: set[AssistantState],
) -> None:
    machine = AssistantStateMachine()
    for state in source_path:
        machine.transition(state)

    assert machine.get() is source
    for target in AssistantState:
        assert machine.can_transition(target) is (
            target is source or target in allowed
        )


def test_reset_returns_directly_to_idle() -> None:
    machine = AssistantStateMachine()
    machine.transition(AssistantState.PROCESSING)
    machine.transition(AssistantState.TOOL_RUNNING)
    machine.transition(AssistantState.STREAMING)

    assert machine.reset() is AssistantState.IDLE
    assert machine.get() is AssistantState.IDLE


def test_machines_do_not_share_state() -> None:
    first = AssistantStateMachine()
    second = AssistantStateMachine()

    first.transition(AssistantState.PROCESSING)

    assert first.get() is AssistantState.PROCESSING
    assert second.get() is AssistantState.IDLE


@pytest.mark.parametrize("target", ["processing", None, 1])
def test_transition_methods_reject_invalid_target_types(target: object) -> None:
    machine = AssistantStateMachine()
    methods: tuple[Callable[[AssistantState], object], ...] = (
        machine.can_transition,
        machine.transition,
    )

    for method in methods:
        with pytest.raises(
            TypeError,
            match="Assistant state target must be an AssistantState",
        ):
            method(target)  # type: ignore[arg-type]

    assert machine.get() is AssistantState.IDLE
