from __future__ import annotations

import boto3
from botocore.client import Config as BotoConfig

from ..application.interfaces import StorageGateway
from ..config import IngestConfig


def create_s3_client(config: IngestConfig):
    return boto3.client(
        "s3",
        endpoint_url=config.storage_endpoint_url,
        region_name=config.storage_region,
        aws_access_key_id=config.storage_access_key,
        aws_secret_access_key=config.storage_secret_key,
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


class S3StorageGateway(StorageGateway):
    def __init__(self, client) -> None:
        self._client = client

    def download(self, bucket: str, object_key: str, destination_path: str) -> None:
        self._client.download_file(bucket, object_key, destination_path)

    def upload(self, bucket: str, object_key: str, source_path: str) -> None:
        self._client.upload_file(source_path, bucket, object_key)


def create_storage_gateway(config: IngestConfig) -> StorageGateway:
    client = create_s3_client(config)
    return S3StorageGateway(client)
