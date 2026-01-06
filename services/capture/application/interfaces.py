from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Mapping

if TYPE_CHECKING:
    from domain.ingest_notification import IngestNotification
    from domain.session import CaptureSession, SessionClip, SessionClipStatus


class IdProvider(Protocol):
    def generate(self) -> str: ...


class MultipartUploadClient(Protocol):
    def initiate_upload(self, object_key: str, content_type: str) -> str: ...

    def generate_part_url(
        self,
        *,
        object_key: str,
        upload_id: str,
        part_no: int,
        expires_in_seconds: int,
    ) -> str: ...

    def complete_upload(
        self, *, object_key: str, upload_id: str, parts: list[tuple[int, str]]
    ) -> str | None: ...

    def generate_put_url(
        self,
        *,
        object_key: str,
        content_type: str,
        expires_in_seconds: int,
    ) -> str: ...


class IngestNotificationRepository(Protocol):
    def save(self, notification: "IngestNotification") -> None: ...

    def list_for_upload(
        self, *, asset_id: str, upload_id: str
    ) -> list["IngestNotification"]: ...


class SessionRepository(Protocol):
    def create(self, session: "CaptureSession") -> "CaptureSession": ...

    def get(self, session_id: str) -> "CaptureSession" | None: ...


class SessionClipRepository(Protocol):
    def create(self, clip: "SessionClip") -> "SessionClip": ...

    def count_for_session(self, session_id: str) -> int: ...

    def update_status(
        self,
        *,
        session_id: str,
        clip_id: str,
        status: "SessionClipStatus",
        metadata: Mapping[str, object] | None = None,
    ) -> "SessionClip": ...


class ClipEventPublisher(Protocol):
    def publish_clip_ready(self, clip: "SessionClip") -> None: ...
