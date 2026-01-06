from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from application.dto import RequestSessionClipUploadCommand
from application.interfaces import (
    MultipartUploadClient,
    SessionClipRepository,
    SessionRepository,
)
from domain.session import SessionClip, SessionClipStatus


class RequestSessionClipUploadUseCase:
    def __init__(
        self,
        *,
        session_repository: SessionRepository,
        clip_repository: SessionClipRepository,
        storage_client: MultipartUploadClient,
        url_ttl: timedelta,
    ) -> None:
        self._session_repository = session_repository
        self._clip_repository = clip_repository
        self._storage_client = storage_client
        self._url_ttl = url_ttl

    def execute(self, command: RequestSessionClipUploadCommand) -> dict[str, object]:
        session = self._session_repository.get(command.session_id)
        if session is None:
            raise ValueError("Session not found")
        now = datetime.now(timezone.utc)
        if session.status != "open":
            raise ValueError("Session is not open")
        if now >= session.expires_at:
            raise ValueError("Session has expired")

        current_clips = self._clip_repository.count_for_session(session.session_id)
        if current_clips >= session.max_clips:
            raise ValueError("Session has reached maximum clips")

        clip_id = uuid.uuid4().hex
        object_key = _build_clip_object_key(
            session.object_prefix, command.filename, clip_id
        )

        clip = SessionClip(
            clip_id=clip_id,
            session_id=session.session_id,
            object_key=object_key,
            status=SessionClipStatus.PENDING_UPLOAD,
            created_at=now,
        )
        self._clip_repository.create(clip)

        expires_at = now + self._url_ttl
        upload_url = self._storage_client.generate_put_url(
            object_key=object_key,
            content_type=command.content_type,
            expires_in_seconds=max(int(self._url_ttl.total_seconds()), 60),
        )

        return {
            "clip_id": clip_id,
            "object_key": object_key,
            "upload_url": upload_url,
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        }


def _sanitize_clip_filename(filename: str, clip_id: str) -> str:
    safe_name = Path(filename or "").name
    if not safe_name:
        safe_name = f"clip-{clip_id}.bin"
    stem = Path(safe_name).stem or f"clip-{clip_id}"
    suffix = Path(safe_name).suffix or ".bin"
    normalized = stem.replace(" ", "_") or f"clip-{clip_id}"
    return f"{normalized}_{clip_id}{suffix}"


def _build_clip_object_key(object_prefix: str, filename: str, clip_id: str) -> str:
    unique_name = _sanitize_clip_filename(filename, clip_id)
    segments = [object_prefix, "clips", unique_name]
    return "/".join(segment for segment in segments if segment)
