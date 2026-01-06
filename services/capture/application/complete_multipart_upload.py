from __future__ import annotations

from application.dto import CompleteMultipartUploadCommand
from application.interfaces import IngestNotificationRepository, MultipartUploadClient
from domain.upload import CompletedMultipartUpload


class CompleteMultipartUploadUseCase:
    def __init__(
        self,
        *,
        multipart_client: MultipartUploadClient,
        notification_repository: IngestNotificationRepository,
    ) -> None:
        self._client = multipart_client
        self._notifications = notification_repository

    def execute(
        self, command: CompleteMultipartUploadCommand
    ) -> CompletedMultipartUpload:
        if not command.parts:
            raise ValueError("At least one part is required to complete an upload")

        observed = self._notifications.list_for_upload(
            asset_id=command.asset_id, upload_id=command.upload_id
        )
        observed_map = {(n.part_no, n.etag): n for n in observed}

        ordered_parts = []
        for part in command.parts:
            key = (part.part_no, part.etag)
            if key not in observed_map:
                raise ValueError(
                    "Part %s with etag %s has not been ingested"
                    % (part.part_no, part.etag)
                )
            ordered_parts.append((part.part_no, part.etag))

        location = self._client.complete_upload(
            object_key=command.object_key,
            upload_id=command.upload_id,
            parts=ordered_parts,
        )

        return CompletedMultipartUpload(
            asset_id=command.asset_id,
            upload_id=command.upload_id,
            object_key=command.object_key,
            location=location,
        )
