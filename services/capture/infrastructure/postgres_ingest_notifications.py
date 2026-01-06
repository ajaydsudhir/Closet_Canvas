from __future__ import annotations

from typing import Mapping, Any, List

from psycopg_pool import ConnectionPool
from psycopg.types.json import Json

from domain.ingest_notification import IngestNotification


class PostgresIngestNotificationRepository:
    def __init__(self, dsn: str) -> None:
        self._pool = ConnectionPool(conninfo=dsn, kwargs={"autocommit": True})
        self._ensure_schema()

    def save(self, notification: IngestNotification) -> None:
        payload = Json(_as_dict(notification.client_meta))
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingest_notifications (
                    asset_id,
                    upload_id,
                    part_no,
                    etag,
                    client_meta,
                    notified_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    notification.asset_id,
                    notification.upload_id,
                    notification.part_no,
                    notification.etag,
                    payload,
                    notification.notified_at,
                ),
            )

    def list_for_upload(
        self, *, asset_id: str, upload_id: str
    ) -> List[IngestNotification]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT asset_id, upload_id, part_no, etag, client_meta, notified_at
                FROM ingest_notifications
                WHERE asset_id = %s AND upload_id = %s
                ORDER BY part_no ASC
                """,
                (asset_id, upload_id),
            ).fetchall()
        return [
            IngestNotification(
                asset_id=row[0],
                upload_id=row[1],
                part_no=row[2],
                etag=row[3],
                client_meta=row[4],
                notified_at=row[5],
            )
            for row in rows
        ]

    def _ensure_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_notifications (
                    id BIGSERIAL PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    upload_id TEXT NOT NULL,
                    part_no INTEGER NOT NULL,
                    etag TEXT NOT NULL,
                    client_meta JSONB NOT NULL,
                    notified_at TIMESTAMPTZ NOT NULL
                )
                """
            )


def _as_dict(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(meta)
