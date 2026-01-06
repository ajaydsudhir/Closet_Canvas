from __future__ import annotations

import boto3
from botocore.client import Config as BotoConfig


class MinioMultipartUploadClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        public_endpoint_url: str | None = None,
        region_name: str,
        bucket_name: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        self._bucket_name = bucket_name
        # Use public endpoint for signature generation if provided
        client_endpoint = public_endpoint_url or endpoint_url
        self._client = boto3.client(
            "s3",
            endpoint_url=client_endpoint,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(
                signature_version="s3v4", s3={"addressing_style": "path"}
            ),
        )

    def initiate_upload(self, object_key: str, content_type: str) -> str:
        response = self._client.create_multipart_upload(
            Bucket=self._bucket_name,
            Key=object_key,
            ContentType=content_type,
        )
        return response["UploadId"]

    def generate_part_url(
        self,
        *,
        object_key: str,
        upload_id: str,
        part_no: int,
        expires_in_seconds: int,
    ) -> str:
        expires = max(expires_in_seconds, 60)
        return self._client.generate_presigned_url(
            ClientMethod="upload_part",
            Params={
                "Bucket": self._bucket_name,
                "Key": object_key,
                "UploadId": upload_id,
                "PartNumber": part_no,
            },
            ExpiresIn=expires,
        )

    def complete_upload(
        self, *, object_key: str, upload_id: str, parts: list[tuple[int, str]]
    ) -> str | None:
        response = self._client.complete_multipart_upload(
            Bucket=self._bucket_name,
            Key=object_key,
            UploadId=upload_id,
            MultipartUpload={
                "Parts": [
                    {"ETag": etag, "PartNumber": part_no} for part_no, etag in parts
                ]
            },
        )
        return response.get("Location")

    def generate_put_url(
        self,
        *,
        object_key: str,
        content_type: str,
        expires_in_seconds: int,
    ) -> str:
        expires = max(expires_in_seconds, 60)
        return self._client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": self._bucket_name,
                "Key": object_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires,
        )
