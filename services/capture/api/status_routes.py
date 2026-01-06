"""REST API routes for session status and recommendations."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Get Redis configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class SessionStatusResponse(BaseModel):
    """Session status response model."""

    session_id: str
    status: str
    message: str | None = None
    progress: float | None = None


class RecommendationItem(BaseModel):
    """Recommendation item model."""

    id: str
    imageUrl: str
    title: str
    brand: str | None = None
    price: float | None = None
    category: str | None = None
    matchScore: float | None = None
    fitScore: float | None = None
    preferenceScore: float | None = None
    size: str | None = None


class RecommendationsResponse(BaseModel):
    """Recommendations response model."""

    items: list[RecommendationItem]
    session_id: str


class RatingRequest(BaseModel):
    """Request model for rating a garment."""

    garment_id: str
    rating: int  # 1-5 scale
    user_id: str | None = None


class SessionStatusManager:
    """Manages WebSocket connections for session status updates."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """Register a new WebSocket connection for a session."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Send a message to all connections for a session."""
        if session_id not in self.active_connections:
            return

        dead_connections = []
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                dead_connections.append(connection)

        # Clean up dead connections
        for connection in dead_connections:
            self.disconnect(session_id, connection)

    async def listen_to_redis(self, session_id: str):
        """Listen to Redis pub/sub for session status updates."""
        redis_client = await aioredis.from_url(
            f"redis://{self.redis_host}:{self.redis_port}",
            encoding="utf-8",
            decode_responses=True,
        )

        try:
            pubsub = redis_client.pubsub()
            channel = f"session:{session_id}:status"
            await pubsub.subscribe(channel)

            logger.info(f"Listening to Redis channel: {channel}")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await self.broadcast_to_session(session_id, data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Redis message: {e}")

                # Stop listening if no connections remain
                if session_id not in self.active_connections:
                    break

        except Exception as e:
            logger.error(f"Error in Redis listener: {e}")
        finally:
            await pubsub.unsubscribe(channel)
            await redis_client.close()


# Global manager instance
status_manager = SessionStatusManager(redis_host=REDIS_HOST, redis_port=REDIS_PORT)


def create_status_router() -> APIRouter:
    """Create router for status and recommendations endpoints."""
    router = APIRouter()

    @router.get(
        "/v1/sessions/{session_id}/status", response_model=SessionStatusResponse
    )
    async def get_session_status(session_id: str) -> SessionStatusResponse:
        """Get the current status of a session."""
        try:
            # Connect to Redis
            redis_client = await aioredis.from_url(
                f"redis://{REDIS_HOST}:{REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
            )

            try:
                # Get session status from Redis
                status_key = f"session:{session_id}:status"
                status_data = await redis_client.hgetall(status_key)

                if not status_data:
                    # Default status if not found
                    return SessionStatusResponse(
                        session_id=session_id, status="idle", message="Session is idle"
                    )

                return SessionStatusResponse(
                    session_id=session_id,
                    status=status_data.get("status", "idle"),
                    message=status_data.get("message"),
                    progress=float(status_data["progress"])
                    if "progress" in status_data
                    else None,
                )

            finally:
                await redis_client.close()

        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            raise HTTPException(status_code=500, detail="Failed to get session status")

    @router.websocket("/v1/sessions/{session_id}/status")
    async def session_status_websocket(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time session status updates."""
        await status_manager.connect(session_id, websocket)

        # Start Redis listener task
        listener_task = asyncio.create_task(status_manager.listen_to_redis(session_id))

        try:
            # Send initial status
            await websocket.send_json(
                {
                    "type": "status",
                    "sessionId": session_id,
                    "status": "idle",
                    "message": "Connected to session",
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

            # Keep connection alive
            while True:
                try:
                    # Receive messages from client (optional, for heartbeat)
                    data = await websocket.receive_text()
                    logger.debug(f"Received from client: {data}")
                except WebSocketDisconnect:
                    break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")

        finally:
            status_manager.disconnect(session_id, websocket)
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    @router.get(
        "/v1/sessions/{session_id}/recommendations",
        response_model=RecommendationsResponse,
    )
    async def get_recommendations(session_id: str) -> RecommendationsResponse:
        """Get recommendations for a completed session."""
        try:
            # Connect to Redis
            redis_client = await aioredis.from_url(
                f"redis://{REDIS_HOST}:{REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
            )

            try:
                # Get recommendations from Redis
                recs_key = f"session:{session_id}:recommendations"
                recommendations_data = await redis_client.lrange(recs_key, 0, -1)

                if not recommendations_data:
                    # Return empty list if no recommendations yet
                    return RecommendationsResponse(items=[], session_id=session_id)

                # Parse recommendations
                items = []
                for item_json in recommendations_data:
                    item_data = json.loads(item_json)
                    items.append(RecommendationItem(**item_data))

                return RecommendationsResponse(items=items, session_id=session_id)

            finally:
                await redis_client.close()

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise HTTPException(status_code=500, detail="Failed to get recommendations")

    @router.post(
        "/v1/sessions/{session_id}/ratings",
        status_code=202,
    )
    async def submit_rating(session_id: str, payload: RatingRequest):
        """Submit a rating for a garment to update preferences.
        
        Ratings are queued and processed by the recommendation worker
        to update the user's preference model.
        """
        try:
            # Validate rating
            if not 1 <= payload.rating <= 5:
                raise HTTPException(
                    status_code=400,
                    detail="Rating must be between 1 and 5"
                )

            # Connect to Redis
            redis_client = await aioredis.from_url(
                f"redis://{REDIS_HOST}:{REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
            )

            try:
                # Store rating in a queue for async processing
                rating_data = {
                    "session_id": session_id,
                    "garment_id": payload.garment_id,
                    "rating": payload.rating,
                    "user_id": payload.user_id,
                }
                
                # Push to rating queue
                ratings_key = f"session:{session_id}:ratings"
                await redis_client.rpush(ratings_key, json.dumps(rating_data))
                await redis_client.expire(ratings_key, 86400)  # 24 hour TTL

                logger.info(
                    f"Queued rating: session={session_id}, garment={payload.garment_id}, rating={payload.rating}"
                )

                return {"status": "accepted", "message": "Rating queued for processing"}

            finally:
                await redis_client.close()

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error submitting rating: {e}")
            raise HTTPException(status_code=500, detail="Failed to submit rating")

    return router
