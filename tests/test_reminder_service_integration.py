from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from friday.database.connection import SessionLocal
from friday.database.models import Reminder
from friday.services.reminder_service import ReminderService

pytestmark = pytest.mark.integration


def test_reminder_service_postgresql_one_time_claim() -> None:
    service = ReminderService()
    reminder_id: int | None = None

    with SessionLocal() as session:
        earliest_due = session.scalar(
            select(Reminder.remind_at)
            .where(
                Reminder.is_enabled.is_(True),
                Reminder.fired_at.is_(None),
                Reminder.remind_at <= datetime.now(UTC),
            )
            .order_by(Reminder.remind_at.asc())
            .limit(1)
        )
    remind_at = (
        earliest_due - timedelta(seconds=1)
        if earliest_due is not None
        else datetime.now(UTC) - timedelta(seconds=1)
    )

    try:
        created = service.create(
            f"Friday reminder integration {uuid4()}",
            remind_at=remind_at,
        )
        reminder_id = created.id
        assert reminder_id in {
            reminder.id for reminder in service.list_upcoming(limit=200)
        }

        first_claim = service.claim_due(now=remind_at, limit=1)
        assert [reminder.id for reminder in first_claim] == [reminder_id]
        assert first_claim[0].is_enabled is False
        assert first_claim[0].fired_at is not None

        retrieved = service.get(reminder_id)
        assert retrieved is not None
        assert retrieved.is_enabled is False
        assert retrieved.fired_at is not None

        second_claim = service.claim_due(now=remind_at, limit=1)
        assert reminder_id not in {reminder.id for reminder in second_claim}
    finally:
        if reminder_id is not None:
            with SessionLocal() as session:
                reminder = session.get(Reminder, reminder_id)
                if reminder is not None:
                    session.delete(reminder)
                    session.commit()
            with SessionLocal() as session:
                assert session.get(Reminder, reminder_id) is None
