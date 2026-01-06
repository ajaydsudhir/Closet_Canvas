from __future__ import annotations

from application.dto import IngestNotificationCommand
from application.interfaces import IngestNotificationRepository
from domain.ingest_notification import IngestNotification


class NotifyIngestUseCase:
    def __init__(self, repository: IngestNotificationRepository) -> None:
        self._repository = repository

    def execute(self, command: IngestNotificationCommand) -> IngestNotification:
        notification = IngestNotification.from_payload(
            asset_id=command.asset_id,
            upload_id=command.upload_id,
            part_no=command.part_no,
            etag=command.etag,
            client_meta=command.client_meta,
        )
        self._repository.save(notification)
        return notification
