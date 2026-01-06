from __future__ import annotations

from application.dto import SessionClipUploadedCommand
from application.interfaces import (
    ClipEventPublisher,
    SessionClipRepository,
    SessionRepository,
)
from domain.session import SessionClipStatus


class SessionClipUploadedUseCase:
    def __init__(
        self,
        *,
        session_repository: SessionRepository,
        clip_repository: SessionClipRepository,
        event_publisher: ClipEventPublisher,
    ) -> None:
        self._session_repository = session_repository
        self._clip_repository = clip_repository
        self._event_publisher = event_publisher

    def execute(self, command: SessionClipUploadedCommand) -> None:
        session = self._session_repository.get(command.session_id)
        if session is None:
            raise ValueError("Session not found")
        if session.status != "open":
            raise ValueError("Session is not open")

        clip = self._clip_repository.update_status(
            session_id=command.session_id,
            clip_id=command.clip_id,
            status=SessionClipStatus.UPLOADED,
            metadata=command.metadata,
        )
        self._event_publisher.publish_clip_ready(clip)
