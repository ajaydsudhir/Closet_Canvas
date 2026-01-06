from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

from application.dto import CreateSessionCommand
from application.interfaces import SessionRepository
from domain.session import CaptureSession


class CreateSessionUseCase:
    def __init__(
        self,
        *,
        repository: SessionRepository,
        object_prefix_root: str,
        default_max_clips: int,
        default_ttl: timedelta,
    ) -> None:
        self._repository = repository
        self._object_prefix_root = object_prefix_root.strip("/")
        self._default_max_clips = default_max_clips
        self._default_ttl = default_ttl

    def execute(self, command: CreateSessionCommand) -> CaptureSession:
        session_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        max_clips = command.max_clips or self._default_max_clips
        ttl_minutes = command.ttl_minutes or int(
            self._default_ttl.total_seconds() // 60
        )
        ttl = timedelta(minutes=ttl_minutes)
        now = datetime.now(timezone.utc)
        expires_at = now + ttl
        object_prefix = "/".join(
            segment
            for segment in [self._object_prefix_root, "sessions", session_id]
            if segment
        )
        session = CaptureSession(
            session_id=session_id,
            token=token,
            status="open",
            object_prefix=object_prefix,
            max_clips=max_clips,
            created_at=now,
            expires_at=expires_at,
            metadata=dict(command.metadata or {}),
        )
        self._repository.create(session)
        return session
