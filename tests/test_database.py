import pytest
from sqlalchemy import select

from friday.database.connection import SessionLocal, engine
from friday.database.models import ToolPermission

pytestmark = pytest.mark.integration


def test_database_connection() -> None:
    with engine.connect() as connection:
        assert connection is not None


def test_can_read_application_tables() -> None:
    with SessionLocal() as session:
        permissions = session.scalars(
            select(ToolPermission)
        ).all()

        assert len(permissions) > 0
