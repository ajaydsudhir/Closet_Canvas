from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from application.dto import CreateMultipartUploadCommand
from application.interfaces import IdProvider, MultipartUploadClient
from domain.upload import MultipartUpload, MultipartUploadPart


class CreateMultipartUploadUseCase:
    def __init__(
        self,
        asset_id_provider: IdProvider,
        multipart_client: MultipartUploadClient,
        max_initial_parts: int,
        expiry_ttl: timedelta,
        object_key_prefix: str,
    ) -> None:
        self._asset_id_provider = asset_id_provider
        self._multipart_client = multipart_client
        self._expiry_ttl = expiry_ttl
        self._max_initial_parts = max_initial_parts
        self._object_key_prefix = object_key_prefix.strip("/")

    def execute(self, command: CreateMultipartUploadCommand) -> MultipartUpload:
        asset_id = self._asset_id_provider.generate()
        object_key = self._build_object_key(
            asset_id=asset_id, filename=command.filename
        )
        upload_id = self._multipart_client.initiate_upload(
            object_key=object_key, content_type=command.content_type
        )
        expires_at = datetime.now(timezone.utc) + self._expiry_ttl

        parts = self._build_parts(
            object_key=object_key, upload_id=upload_id, max_parts=command.max_parts
        )

        return MultipartUpload(
            asset_id=asset_id,
            upload_id=upload_id,
            object_key=object_key,
            parts=parts,
            expires_at=expires_at,
        )

    def _build_parts(
        self, object_key: str, upload_id: str, max_parts: int
    ) -> List[MultipartUploadPart]:
        parts_to_issue = min(self._max_initial_parts, max_parts)
        expires_in_seconds = max(int(self._expiry_ttl.total_seconds()), 60)
        return [
            MultipartUploadPart(
                part_no=part_no,
                url=self._multipart_client.generate_part_url(
                    object_key=object_key,
                    upload_id=upload_id,
                    part_no=part_no,
                    expires_in_seconds=expires_in_seconds,
                ),
            )
            for part_no in range(1, parts_to_issue + 1)
        ]

    def _build_object_key(self, asset_id: str, filename: str) -> str:
        safe_name = Path(filename).name or "upload.bin"
        segments = [
            segment
            for segment in [self._object_key_prefix, asset_id, safe_name]
            if segment
        ]
        return "/".join(segments)
