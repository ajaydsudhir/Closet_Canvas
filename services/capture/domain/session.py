from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional
from enum import Enum


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    token: str
    status: str
    object_prefix: str
    max_clips: int
    created_at: datetime
    expires_at: datetime
    metadata: Mapping[str, object]


class SessionClipStatus(str, Enum):
    FAILED = "failed"
    PENDING_UPLOAD = "pending_upload"
    PREPROCESSED = "preprocessed"
    PREPROCESSING = "preprocessing"
    UPLOADED = "uploaded"


@dataclass(frozen=True)
class SessionClip:
    clip_id: str
    session_id: str
    object_key: str
    status: SessionClipStatus
    created_at: datetime
    uploaded_at: Optional[datetime] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
