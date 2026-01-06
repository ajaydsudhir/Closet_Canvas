from __future__ import annotations

from redis import Redis
from rq import Queue, Worker as RQWorker

from ..config import IngestConfig


def create_redis_connection(config: IngestConfig) -> Redis:
    return Redis(host=config.redis_host, port=config.redis_port, db=config.redis_db)


def create_queue(config: IngestConfig, queue_name: str | None = None) -> Queue:
    redis_conn = create_redis_connection(config)
    # Use longer timeout for gating queue (model loading + inference on CPU)
    # Recommendation queue is fast since it's just calculations
    if queue_name == "gating":
        default_timeout = 1000
    elif queue_name == "recommendation":
        default_timeout = 300
    else:
        default_timeout = 180
    return Queue(
        queue_name or config.redis_queue_name,
        connection=redis_conn,
        default_timeout=default_timeout,
    )


def create_worker(config: IngestConfig, queue_name: str | None = None) -> RQWorker:
    redis_conn = create_redis_connection(config)
    queue = Queue(queue_name or config.redis_queue_name, connection=redis_conn)
    return RQWorker([queue], connection=redis_conn)
