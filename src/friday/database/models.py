from datetime import datetime
from typing import Any

from sqlalchemy import (
    REAL,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from friday.database.connection import Base

# ============================================================
# App settings
# ============================================================


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )

    value_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# ============================================================
# Conversations
# ============================================================


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    title: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    mode: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'local'"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


# ============================================================
# Messages
# ============================================================


class Message(Base):
    __tablename__ = "messages"

    __table_args__ = (
        CheckConstraint(
            "role IN ('system','user','assistant','tool')",
            name="messages_role_check",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Database column is named "metadata".
    # Python attribute is metadata_json because SQLAlchemy already
    # uses the name "metadata" internally.
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages",
    )

    memories: Mapped[list["Memory"]] = relationship(
        back_populates="source_message",
    )


# ============================================================
# Long-term memory
# ============================================================


class Memory(Base):
    __tablename__ = "memories"

    __table_args__ = (
        CheckConstraint(
            "importance BETWEEN 1 AND 10",
            name="memories_importance_check",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    memory_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'fact'"),
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    importance: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        server_default=text("5"),
    )

    source_message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey(
            "messages.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("TRUE"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    source_message: Mapped["Message | None"] = relationship(
        back_populates="memories",
    )


# ============================================================
# Tasks
# ============================================================


class Task(Base):
    __tablename__ = "tasks"

    __table_args__ = (
        CheckConstraint(
            "status IN ('open','in_progress','done','cancelled')",
            name="tasks_status_check",
        ),
        CheckConstraint(
            "priority BETWEEN 1 AND 5",
            name="tasks_priority_check",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'open'"),
    )

    priority: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        server_default=text("3"),
    )

    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    reminders: Mapped[list["Reminder"]] = relationship(
        back_populates="task",
    )


# ============================================================
# Reminders
# ============================================================


class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    remind_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    recurrence: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    task_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey(
            "tasks.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("TRUE"),
    )

    fired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    task: Mapped["Task | None"] = relationship(
        back_populates="reminders",
    )


# ============================================================
# Tool permissions
# ============================================================


class ToolPermission(Base):
    __tablename__ = "tool_permissions"

    __table_args__ = (
        CheckConstraint(
            "permission_mode IN ('deny','ask','allow')",
            name="tool_permissions_permission_mode_check",
        ),
    )

    tool_name: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )

    permission_mode: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'ask'"),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


# ============================================================
# System metrics
# ============================================================


class SystemMetric(Base):
    __tablename__ = "system_metrics"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    cpu_percent: Mapped[float | None] = mapped_column(
        REAL,
        nullable=True,
    )

    ram_percent: Mapped[float | None] = mapped_column(
        REAL,
        nullable=True,
    )

    gpu_percent: Mapped[float | None] = mapped_column(
        REAL,
        nullable=True,
    )

    gpu_memory_percent: Mapped[float | None] = mapped_column(
        REAL,
        nullable=True,
    )

    sampled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# ============================================================
# Audit log
# ============================================================


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    event_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    action: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    success: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# ============================================================
# Existing PostgreSQL indexes
# ============================================================


Index(
    "idx_messages_conversation_created",
    Message.conversation_id,
    Message.created_at,
)

Index(
    "idx_memories_active_importance",
    Memory.is_active,
    Memory.importance.desc(),
)

Index(
    "idx_tasks_status_due",
    Task.status,
    Task.due_at,
)

Index(
    "idx_reminders_enabled_time",
    Reminder.is_enabled,
    Reminder.remind_at,
)

Index(
    "idx_metrics_sampled_at",
    SystemMetric.sampled_at.desc(),
)

Index(
    "idx_audit_created_at",
    AuditLog.created_at.desc(),
)
