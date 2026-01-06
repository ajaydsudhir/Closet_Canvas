"""Utilities for publishing session status updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

SessionStatus = Literal[
    "idle",
    "recording",
    "gating",
    "smpl",
    "finishing",
    "recommending",
    "complete",
    "error",
]


class SessionStatusPublisher:
    """Publishes session status updates to Redis for WebSocket broadcasting."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port

    async def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        message: str | None = None,
        progress: float | None = None,
    ) -> None:
        """Update session status and broadcast to WebSocket listeners."""
        try:
            redis_client = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}",
                encoding="utf-8",
                decode_responses=True,
            )

            try:
                # Update session status in Redis hash
                status_key = f"session:{session_id}:status"
                status_data = {"status": status}
                if message:
                    status_data["message"] = message
                if progress is not None:
                    status_data["progress"] = str(progress)

                await redis_client.hset(status_key, mapping=status_data)
                await redis_client.expire(status_key, 3600)  # Expire after 1 hour

                # Publish to WebSocket channel
                channel = f"session:{session_id}:status"
                message_data = {
                    "type": "status",
                    "sessionId": session_id,
                    "status": status,
                    "message": message,
                    "progress": progress,
                    "timestamp": str(asyncio.get_event_loop().time()),
                }
                await redis_client.publish(channel, json.dumps(message_data))

                logger.info(
                    f"Published status update for session {session_id}: {status}"
                )

            finally:
                await redis_client.close()

        except Exception as e:
            logger.error(f"Error publishing status update: {e}")

    async def store_recommendations(
        self, session_id: str, recommendations: list[dict]
    ) -> None:
        """Store recommendations in Redis for retrieval."""
        try:
            redis_client = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}",
                encoding="utf-8",
                decode_responses=True,
            )

            try:
                recs_key = f"session:{session_id}:recommendations"

                # Clear existing recommendations
                await redis_client.delete(recs_key)

                # Store new recommendations
                if recommendations:
                    serialized_recs = [json.dumps(rec) for rec in recommendations]
                    await redis_client.rpush(recs_key, *serialized_recs)
                    await redis_client.expire(recs_key, 3600)  # Expire after 1 hour

                logger.info(
                    f"Stored {len(recommendations)} recommendations for session {session_id}"
                )

            finally:
                await redis_client.close()

        except Exception as e:
            logger.error(f"Error storing recommendations: {e}")
