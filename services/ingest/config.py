from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestConfig:
    storage_endpoint_url: str
    storage_region: str
    storage_access_key: str
    storage_secret_key: str
    redis_host: str
    redis_port: int
    redis_db: int
    redis_queue_name: str
    redis_channel: str
    remux_target_extension: str
    header_cache_ttl_seconds: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> IngestConfig:
    return IngestConfig(
        storage_endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        storage_region=os.getenv("MINIO_REGION", "us-east-1"),
        storage_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        storage_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_queue_name=os.getenv("REDIS_QUEUE_NAME", "video"),
        redis_channel=os.getenv("REDIS_CHANNEL", "clip_ready"),
        remux_target_extension=os.getenv("REMUX_TARGET_EXTENSION", ".mp4"),
        header_cache_ttl_seconds=int(os.getenv("HEADER_CACHE_TTL_SECONDS", "3600")),
    )
