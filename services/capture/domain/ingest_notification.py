from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True)
class IngestNotification:
    asset_id: str
    upload_id: str
    part_no: int
    etag: str
    client_meta: Mapping[str, Any]
    notified_at: datetime

    @classmethod
    def from_payload(
        cls,
        *,
        asset_id: str,
        upload_id: str,
        part_no: int,
        etag: str,
        client_meta: Mapping[str, Any],
    ) -> "IngestNotification":
        return cls(
            asset_id=asset_id,
            upload_id=upload_id,
            part_no=part_no,
            etag=etag,
            client_meta=client_meta,
            notified_at=datetime.now(timezone.utc),
        )
