from __future__ import annotations

import json
from typing import Callable

import redis

from .config import IngestConfig


def listen_for_clip_ready(
    config: IngestConfig,
    enqueue_fn: Callable[[dict[str, object]], object],
    stop_event=None,
) -> None:
    client = redis.Redis(
        host=config.redis_host, port=config.redis_port, db=config.redis_db
    )
    pubsub = client.pubsub()
    pubsub.subscribe(config.redis_channel)
    print(f"Listening for clip events on channel {config.redis_channel!r}")
    for message in pubsub.listen():
        if stop_event and stop_event.is_set():
            break
        if message.get("type") != "message":
            continue
        try:
            payload = json.loads(message["data"])
        except (TypeError, json.JSONDecodeError):
            continue
        bucket = payload.get("bucket")
        key = payload.get("object_key")
        session_id = payload.get("session_id")
        clip_id = payload.get("clip_id")
        if not bucket or not key or not session_id or not clip_id:
            continue
        print(f"New clip event: {bucket}/{key}")
        enqueue_fn(
            {
                "session_id": session_id,
                "clip_id": clip_id,
                "bucket": bucket,
                "object_key": key,
                "metadata": payload.get("metadata", {}),
            }
        )
