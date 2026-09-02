"""Small durable state store for Control Plane-owned audit and sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "control_audit_events"
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(128), index=True)
    target: Mapped[str] = mapped_column(String(512), index=True)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reason: Mapped[str] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    result: Mapped[str] = mapped_column(String(64), index=True)


class ManualEngine(Base):
    __tablename__ = "control_manual_engines"
    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    management_url: Mapped[str] = mapped_column(String(1024))
    token_env: Mapped[str | None] = mapped_column(String(255))
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AgentSession(Base):
    __tablename__ = "control_agent_sessions"
    resume_token: Mapped[str] = mapped_column(String(128), primary_key=True)
    actor: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[str] = mapped_column(String(64))
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    seen_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ControlStore:
    def __init__(self, database_url: str) -> None:
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(database_url, **kwargs)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def audit(self, **values: Any) -> dict[str, Any]:
        with self.sessions() as session:
            values["before"] = _jsonable(values.get("before"))
            values["after"] = _jsonable(values.get("after"))
            event = AuditEvent(**values)
            session.add(event)
            session.commit()
            return self._row(event)

    def audit_events(self, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        with self.sessions() as session:
            rows = session.scalars(select(AuditEvent).order_by(AuditEvent.sequence.desc()).offset(offset).limit(limit)).all()
            return {"items": [self._row(row) for row in rows], "limit": limit, "offset": offset}

    def manual_engines(self) -> list[dict[str, Any]]:
        with self.sessions() as session:
            return [self._row(row) for row in session.scalars(select(ManualEngine).order_by(ManualEngine.name)).all()]

    def put_engine(self, values: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        with self.sessions() as session:
            row = session.get(ManualEngine, values["name"])
            before = self._row(row) if row else None
            if row is None:
                row = ManualEngine(**values)
                session.add(row)
            else:
                for key, value in values.items():
                    setattr(row, key, value)
            session.commit()
            return before, self._row(row)

    def delete_engine(self, name: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(ManualEngine, name)
            if row is None:
                return None
            before = self._row(row)
            session.delete(row)
            session.commit()
            return before

    def get_agent_session(self, token: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(AgentSession, token)
            return self._row(row) if row else None

    def put_agent_session(self, values: dict[str, Any]) -> None:
        with self.sessions() as session:
            row = session.get(AgentSession, values["resume_token"])
            if row is None:
                session.add(AgentSession(**values))
            else:
                row.events = values["events"]
                row.seen_message_ids = values["seen_message_ids"]
                row.updated_at = utcnow()
            session.commit()

    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        return {column.key: getattr(row, column.key) for column in row.__mapper__.column_attrs}


def _jsonable(value: Any) -> Any:
    """Convert ORM snapshots to values accepted by SQLite and PostgreSQL JSON."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
