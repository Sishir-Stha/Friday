from datetime import datetime
from typing import Any

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from friday.database.models import Conversation, Memory, Message, Reminder, Task


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_conversation(
        self,
        title: str | None = None,
        mode: str = "local",
    ) -> Conversation:
        conversation = Conversation(
            title=title,
            mode=mode,
        )

        self.session.add(conversation)
        self.session.flush()

        return conversation

    def get_conversation(
        self,
        conversation_id: int,
    ) -> Conversation | None:
        return self.session.get(
            Conversation,
            conversation_id,
        )

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        model: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            metadata_json=metadata_json or {},
        )

        self.session.add(message)
        self.session.flush()

        return message

    def get_recent_messages(
        self,
        conversation_id: int,
        limit: int = 20,
    ) -> list[Message]:
        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )

        messages = list(
            self.session.scalars(statement).all()
        )

        messages.reverse()

        return messages


class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_memory(
        self,
        *,
        content: str,
        memory_type: str = "fact",
        importance: int = 5,
        source_message_id: int | None = None,
    ) -> Memory:
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            source_message_id=source_message_id,
            is_active=True,
        )

        self.session.add(memory)
        self.session.flush()

        return memory

    def get_memory(self, memory_id: int) -> Memory | None:
        return self.session.get(Memory, memory_id)

    def list_active_memories(
        self,
        *,
        limit: int = 20,
        memory_type: str | None = None,
    ) -> list[Memory]:
        statement = select(Memory).where(
            Memory.is_active.is_(True),
        )

        if memory_type is not None:
            statement = statement.where(
                Memory.memory_type == memory_type,
            )

        statement = statement.order_by(
            Memory.importance.desc(),
            Memory.updated_at.desc(),
            Memory.id.desc(),
        ).limit(limit)

        return list(self.session.scalars(statement).all())

    def deactivate_memory(
        self,
        memory_id: int,
    ) -> Memory | None:
        memory = self.get_memory(memory_id)

        if memory is None:
            return None

        memory.is_active = False
        self.session.flush()

        return memory

    def reactivate_memory(
        self,
        memory_id: int,
    ) -> Memory | None:
        memory = self.get_memory(memory_id)

        if memory is None:
            return None

        memory.is_active = True
        self.session.flush()

        return memory


class TaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_task(
        self,
        *,
        title: str,
        description: str | None,
        priority: int,
        due_at: datetime | None,
    ) -> Task:
        task = Task(
            title=title,
            description=description,
            status="open",
            priority=priority,
            due_at=due_at,
        )
        self.session.add(task)
        self.session.flush()
        return task

    def get_task(self, task_id: int) -> Task | None:
        return self.session.get(Task, task_id)

    def list_tasks(
        self,
        *,
        status: str | None,
        limit: int,
    ) -> list[Task]:
        statement = select(Task)
        if status is not None:
            statement = statement.where(Task.status == status)

        unfinished_first = case(
            (Task.status.in_(("open", "in_progress")), 0),
            else_=1,
        )
        statement = statement.order_by(
            unfinished_first,
            Task.due_at.asc().nulls_last(),
            Task.priority.desc(),
            Task.created_at.desc(),
            Task.id.desc(),
        ).limit(limit)
        return list(self.session.scalars(statement).all())

    def update_task(
        self,
        task_id: int,
        *,
        title: str,
        description: str | None,
        priority: int,
        due_at: datetime | None,
    ) -> Task | None:
        task = self.get_task(task_id)
        if task is None:
            return None

        task.title = title
        task.description = description
        task.priority = priority
        task.due_at = due_at
        self.session.flush()
        return task

    def set_task_status(
        self,
        task_id: int,
        *,
        status: str,
        completed_at: datetime | None,
    ) -> Task | None:
        task = self.get_task(task_id)
        if task is None:
            return None

        task.status = status
        task.completed_at = completed_at
        self.session.flush()
        return task


class ReminderRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_reminder(
        self,
        *,
        title: str,
        remind_at: datetime,
        recurrence: str | None,
        task_id: int | None,
    ) -> Reminder:
        reminder = Reminder(
            title=title,
            remind_at=remind_at,
            recurrence=recurrence,
            task_id=task_id,
            is_enabled=True,
        )
        self.session.add(reminder)
        self.session.flush()
        return reminder

    def get_reminder(self, reminder_id: int) -> Reminder | None:
        return self.session.get(Reminder, reminder_id)

    def task_exists(self, task_id: int) -> bool:
        return self.session.get(Task, task_id) is not None

    def list_upcoming(self, *, limit: int) -> list[Reminder]:
        statement = (
            select(Reminder)
            .where(
                Reminder.is_enabled.is_(True),
                Reminder.fired_at.is_(None),
            )
            .order_by(Reminder.remind_at.asc(), Reminder.id.asc())
            .limit(limit)
        )
        return list(self.session.scalars(statement).all())

    def set_enabled(
        self,
        reminder_id: int,
        *,
        enabled: bool,
    ) -> Reminder | None:
        reminder = self.get_reminder(reminder_id)
        if reminder is None:
            return None

        reminder.is_enabled = enabled
        self.session.flush()
        return reminder

    def claim_due(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> list[Reminder]:
        statement = (
            select(Reminder)
            .where(
                Reminder.is_enabled.is_(True),
                Reminder.remind_at <= now,
                Reminder.fired_at.is_(None),
            )
            .order_by(Reminder.remind_at.asc(), Reminder.id.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        reminders = list(self.session.scalars(statement).all())
        for reminder in reminders:
            reminder.fired_at = now
            reminder.is_enabled = False
        self.session.flush()
        return reminders
