from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class MultipartUploadPart:
    part_no: int
    url: str


@dataclass(frozen=True)
class MultipartUpload:
    asset_id: str
    upload_id: str
    object_key: str
    parts: List[MultipartUploadPart]
    expires_at: datetime


@dataclass(frozen=True)
class CompletedMultipartUpload:
    asset_id: str
    upload_id: str
    object_key: str
    location: Optional[str]
