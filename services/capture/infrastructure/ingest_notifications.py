from __future__ import annotations

from threading import Lock
from typing import List

from domain.ingest_notification import IngestNotification


class InMemoryIngestNotificationRepository:
    def __init__(self) -> None:
        self._lock = Lock()
        self._notifications: List[IngestNotification] = []

    def save(self, notification: IngestNotification) -> None:
        with self._lock:
            self._notifications.append(notification)

    def list_for_upload(
        self, *, asset_id: str, upload_id: str
    ) -> List[IngestNotification]:
        with self._lock:
            return [
                n
                for n in self._notifications
                if n.asset_id == asset_id and n.upload_id == upload_id
            ]
