from __future__ import annotations

from redis import Redis

from services.ingest.application.interfaces import SegmentHeaderCache
from services.ingest.config import IngestConfig
from services.ingest.infrastructure.queue import create_redis_connection


class RedisSegmentHeaderCache(SegmentHeaderCache):
    def __init__(self, client: Redis, *, ttl_seconds: int = 3600) -> None:
        self._client = client
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"ingest:segment-header:{session_id}"

    def set(self, session_id: str, header: bytes) -> None:
        self._client.setex(self._key(session_id), self._ttl, header)

    def get(self, session_id: str) -> bytes | None:
        value = self._client.get(self._key(session_id))
        return bytes(value) if value is not None else None


def create_header_cache(config: IngestConfig) -> SegmentHeaderCache:
    redis_conn = create_redis_connection(config)
    return RedisSegmentHeaderCache(
        redis_conn, ttl_seconds=config.header_cache_ttl_seconds
    )
