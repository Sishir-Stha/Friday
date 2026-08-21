from dataclasses import FrozenInstanceError

import pytest

from friday.services.context_manager import ContextManager, ContextSnapshot


def test_context_starts_empty() -> None:
    manager = ContextManager()

    assert manager.get() == ContextSnapshot()


def test_update_one_field() -> None:
    manager = ContextManager()

    snapshot = manager.update(active_module="system")

    assert snapshot.active_module == "system"
    assert snapshot.active_window is None
    assert snapshot.selected_item is None
    assert snapshot.current_conversation_id is None
    assert snapshot.recent_tool_result is None


def test_update_preserves_unspecified_fields() -> None:
    manager = ContextManager()
    manager.update(
        active_module="system",
        selected_item="gpu",
    )

    snapshot = manager.update(active_window="Task Manager")

    assert snapshot.active_module == "system"
    assert snapshot.active_window == "Task Manager"
    assert snapshot.selected_item == "gpu"


def test_update_can_explicitly_clear_one_field() -> None:
    manager = ContextManager()
    manager.update(
        active_module="system",
        selected_item="gpu",
    )

    snapshot = manager.update(selected_item=None)

    assert snapshot.active_module == "system"
    assert snapshot.selected_item is None


def test_string_values_are_normalized() -> None:
    manager = ContextManager()

    snapshot = manager.update(
        active_module="  system  ",
        active_window="   ",
        selected_item="  gpu ",
        recent_tool_result="  complete  ",
    )

    assert snapshot.active_module == "system"
    assert snapshot.active_window is None
    assert snapshot.selected_item == "gpu"
    assert snapshot.recent_tool_result == "complete"


def test_positive_conversation_id_is_valid() -> None:
    manager = ContextManager()

    snapshot = manager.update(current_conversation_id=1)

    assert snapshot.current_conversation_id == 1


@pytest.mark.parametrize("conversation_id", [0, -1, True, False])
def test_invalid_conversation_id_is_rejected(
    conversation_id: int,
) -> None:
    manager = ContextManager()

    with pytest.raises(
        ValueError,
        match="Conversation ID must be a positive integer",
    ):
        manager.update(current_conversation_id=conversation_id)


def test_clear_resets_all_fields() -> None:
    manager = ContextManager()
    manager.update(
        active_module="system",
        active_window="Task Manager",
        selected_item="gpu",
        current_conversation_id=1,
        recent_tool_result="complete",
    )

    snapshot = manager.clear()

    assert snapshot == ContextSnapshot()
    assert manager.get() == ContextSnapshot()


def test_snapshots_are_immutable_and_not_mutated_by_updates() -> None:
    manager = ContextManager()
    first = manager.get()

    second = manager.update(active_module="system")

    assert first.active_module is None
    assert second.active_module == "system"
    assert first is not second

    with pytest.raises(FrozenInstanceError):
        first.active_module = "tasks"  # type: ignore[misc]


def test_manager_instances_do_not_share_state() -> None:
    first_manager = ContextManager()
    second_manager = ContextManager()

    first_manager.update(active_module="system")

    assert first_manager.get().active_module == "system"
    assert second_manager.get() == ContextSnapshot()
