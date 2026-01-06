from dataclasses import dataclass
from typing import Mapping
from typing import List


@dataclass(frozen=True)
class CreateMultipartUploadCommand:
    filename: str
    content_type: str
    part_size_bytes: int
    max_parts: int


@dataclass(frozen=True)
class IngestNotificationCommand:
    asset_id: str
    upload_id: str
    part_no: int
    etag: str
    client_meta: dict[str, object]


@dataclass(frozen=True)
class CompletionPart:
    part_no: int
    etag: str


@dataclass(frozen=True)
class CompleteMultipartUploadCommand:
    asset_id: str
    upload_id: str
    object_key: str
    parts: List[CompletionPart]


@dataclass(frozen=True)
class CreateSessionCommand:
    max_clips: int | None = None
    ttl_minutes: int | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RequestSessionClipUploadCommand:
    session_id: str
    filename: str
    content_type: str


@dataclass(frozen=True)
class SessionClipUploadedCommand:
    session_id: str
    clip_id: str
    metadata: Mapping[str, object]
