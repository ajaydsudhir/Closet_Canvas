from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from sqlalchemy import Column, DateTime, Integer, String, JSON, func

from application.interfaces import SessionRepository, SessionClipRepository
from domain.session import CaptureSession, SessionClip, SessionClipStatus
from infrastructure.db import Base


class SessionRecord(Base):
    __tablename__ = "capture_sessions"

    session_id = Column(String, primary_key=True)
    token = Column(String, nullable=False)
    status = Column(String, nullable=False)
    object_prefix = Column(String, nullable=False)
    max_clips = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    meta = Column(JSON, nullable=False, default=dict)


class SessionClipRecord(Base):
    __tablename__ = "session_clips"

    clip_id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    object_key = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=True)
    meta = Column(JSON, nullable=False, default=dict)


class PostgresSessionRepository(SessionRepository):
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def create(self, session: CaptureSession) -> CaptureSession:
        record = SessionRecord(
            session_id=session.session_id,
            token=session.token,
            status=session.status,
            object_prefix=session.object_prefix,
            max_clips=session.max_clips,
            created_at=session.created_at,
            expires_at=session.expires_at,
            meta=dict(session.metadata),
        )
        with self._session_factory() as db:
            db.add(record)
            db.commit()
        return session

    def get(self, session_id: str) -> CaptureSession | None:
        with self._session_factory() as db:
            record = db.get(SessionRecord, session_id)
            if record is None:
                return None
            return CaptureSession(
                session_id=record.session_id,
                token=record.token,
                status=record.status,
                object_prefix=record.object_prefix,
                max_clips=record.max_clips,
                created_at=record.created_at,
                expires_at=record.expires_at,
                metadata=dict(record.meta or {}),
            )


class PostgresSessionClipRepository(SessionClipRepository):
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def create(self, clip: SessionClip) -> SessionClip:
        record = SessionClipRecord(
            clip_id=clip.clip_id,
            session_id=clip.session_id,
            object_key=clip.object_key,
            status=clip.status.value
            if isinstance(clip.status, SessionClipStatus)
            else clip.status,
            created_at=clip.created_at,
        )
        with self._session_factory() as db:
            db.add(record)
            db.commit()
        return clip

    def count_for_session(self, session_id: str) -> int:
        with self._session_factory() as db:
            return (
                db.query(func.count(SessionClipRecord.clip_id))
                .filter(SessionClipRecord.session_id == session_id)
                .scalar()
            )

    def update_status(
        self,
        *,
        session_id: str,
        clip_id: str,
        status: SessionClipStatus,
        metadata: Mapping[str, object] | None = None,
    ) -> SessionClip:
        with self._session_factory() as db:
            record = (
                db.query(SessionClipRecord)
                .filter(
                    SessionClipRecord.session_id == session_id,
                    SessionClipRecord.clip_id == clip_id,
                )
                .one_or_none()
            )
            if record is None:
                raise ValueError("Clip not found")
            record.status = status.value
            if metadata is not None:
                record.meta = dict(metadata)
            if status is SessionClipStatus.UPLOADED:
                record.uploaded_at = datetime.now(timezone.utc)
            db.commit()
            return SessionClip(
                clip_id=record.clip_id,
                session_id=record.session_id,
                object_key=record.object_key,
                status=SessionClipStatus(record.status),
                created_at=record.created_at,
                uploaded_at=record.uploaded_at,
                metadata=dict(record.meta or {}),
            )
