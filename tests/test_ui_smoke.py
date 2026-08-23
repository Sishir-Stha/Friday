from threading import Event

from test_ui_main_window import FakeConversation, _runtime, _wait

from friday.ui.main_window import FridayMainWindow


def test_offscreen_window_constructs_shows_and_closes(qapp: object) -> None:
    window = FridayMainWindow(_runtime(), start_background_workers=False)

    window.show()
    qapp.processEvents()  # type: ignore[attr-defined]
    assert window.isVisible()

    window.close()
    qapp.processEvents()  # type: ignore[attr-defined]
    assert not window.isVisible()


def test_offscreen_close_during_active_worker_is_deferred(qapp: object) -> None:
    gate = Event()
    conversation = FakeConversation(gate=gate)
    window = FridayMainWindow(
        _runtime(conversation=conversation),
        start_background_workers=False,
    )
    window.show()
    window.chat_widget.composer.setPlainText("hello")
    window.chat_widget.send_current_message()
    _wait(qapp, conversation.started.is_set)

    window.close()
    assert window.isVisible()
    assert window.chat_widget.has_active_worker

    gate.set()
    _wait(qapp, lambda: not window.isVisible())
    assert not window.chat_widget.has_active_worker
