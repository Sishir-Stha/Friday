from datetime import UTC, datetime
from time import monotonic, sleep

from PySide6.QtWidgets import QSystemTrayIcon, QWidget

from friday.services.reminder_service import ReminderSnapshot
from friday.ui.reminder_notifier import (
    REMINDER_POLL_INTERVAL_MS,
    ReminderNotifier,
)


class FakeReminderService:
    def __init__(self) -> None:
        self.calls = 0

    def claim_due(self) -> list[ReminderSnapshot]:
        self.calls += 1
        return [
            ReminderSnapshot(
                id=1,
                title="Stand up",
                remind_at=datetime.now(UTC),
                recurrence=None,
                task_id=None,
                is_enabled=False,
                fired_at=datetime.now(UTC),
                created_at=datetime.now(UTC),
            )
        ]


def test_notifier_claims_on_worker_and_uses_fallback(
    qapp: object,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        QSystemTrayIcon,
        "isSystemTrayAvailable",
        lambda: False,
    )
    service = FakeReminderService()
    parent = QWidget()
    notifier = ReminderNotifier(service, parent, auto_start=False)
    received: list[str] = []
    notifier.fallback_notification.connect(received.append)

    notifier.poll_now()
    deadline = monotonic() + 2
    while notifier._thread is not None and monotonic() < deadline:
        qapp.processEvents()  # type: ignore[attr-defined]
        sleep(0.005)

    assert notifier._thread is None
    assert service.calls == 1
    assert received == ["Stand up"]
    assert notifier.timer.interval() == REMINDER_POLL_INTERVAL_MS
    notifier.stop()
    parent.close()
