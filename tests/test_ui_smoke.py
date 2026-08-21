from test_ui_main_window import _runtime

from friday.ui.main_window import FridayMainWindow


def test_offscreen_window_constructs_shows_and_closes(qapp: object) -> None:
    window = FridayMainWindow(_runtime(), start_background_workers=False)

    window.show()
    qapp.processEvents()  # type: ignore[attr-defined]
    assert window.isVisible()

    window.close()
    qapp.processEvents()  # type: ignore[attr-defined]
    assert not window.isVisible()
