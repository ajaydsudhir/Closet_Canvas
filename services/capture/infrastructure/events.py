from __future__ import annotations

import json
import logging
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from application.interfaces import ClipEventPublisher
from domain.session import SessionClip

LOGGER = logging.getLogger(__name__)


class LoggingClipEventPublisher(ClipEventPublisher):
    def publish_clip_ready(self, clip: SessionClip) -> None:
        LOGGER.info(
            {
                "event": "video",
                "session_id": clip.session_id,
                "clip_id": clip.clip_id,
                "object_key": clip.object_key,
            }
        )


class RedisClipEventPublisher(ClipEventPublisher):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        db: int,
        channel: str,
        bucket_name: str,
    ) -> None:
        self._redis = Redis(host=host, port=port, db=db, decode_responses=False)
        self._channel = channel
        self._bucket = bucket_name

    def publish_clip_ready(self, clip: SessionClip) -> None:
        payload: dict[str, Any] = {
            "event": "video",
            "session_id": clip.session_id,
            "clip_id": clip.clip_id,
            "bucket": self._bucket,
            "object_key": clip.object_key,
            "metadata": dict(clip.metadata),
        }
        try:
            self._redis.publish(self._channel, json.dumps(payload))
        except RedisError as exc:
            LOGGER.error(
                "Failed to publish clip event for %s/%s: %s",
                clip.session_id,
                clip.clip_id,
                exc,
            )
