from __future__ import annotations

from datetime import timedelta

from fastapi import FastAPI

from api.routes import create_router
from api.status_routes import create_status_router
from application.create_multipart_upload import CreateMultipartUploadUseCase
from application.notify_ingest import NotifyIngestUseCase
from application.complete_multipart_upload import CompleteMultipartUploadUseCase
from application.create_session import CreateSessionUseCase
from application.request_session_clip_upload import RequestSessionClipUploadUseCase
from application.session_clip_uploaded import SessionClipUploadedUseCase
from application.login_user import LoginUserUseCase
from application.save_user_measurements import SaveUserMeasurementsUseCase
from config import CaptureConfig, load_config
from infrastructure.ids import TokenIdProvider
from infrastructure.postgres_ingest_notifications import (
    PostgresIngestNotificationRepository,
)
from infrastructure.minio_uploads import MinioMultipartUploadClient
from infrastructure.sessions import (
    PostgresSessionRepository,
    PostgresSessionClipRepository,
)
from infrastructure.users import (
    PostgresUserRepository,
    PostgresUserMeasurementsRepository,
)
from infrastructure.events import RedisClipEventPublisher
from infrastructure.db import create_session_factory


def build_app(config: CaptureConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"message": "pong"}

    asset_id_provider = TokenIdProvider(
        prefix=cfg.asset_id_prefix, length=cfg.asset_id_length
    )
    multipart_client = MinioMultipartUploadClient(
        endpoint_url=cfg.storage_endpoint_url,
        public_endpoint_url=cfg.storage_public_endpoint_url,
        region_name=cfg.storage_region,
        bucket_name=cfg.storage_bucket,
        access_key=cfg.storage_access_key,
        secret_key=cfg.storage_secret_key,
    )

    create_upload_use_case = CreateMultipartUploadUseCase(
        asset_id_provider=asset_id_provider,
        multipart_client=multipart_client,
        max_initial_parts=cfg.max_initial_parts,
        expiry_ttl=timedelta(hours=cfg.upload_expiry_hours),
        object_key_prefix=cfg.storage_object_prefix,
    )

    ingest_notification_repo = PostgresIngestNotificationRepository(
        dsn=cfg.database_dsn
    )
    notify_ingest_use_case = NotifyIngestUseCase(repository=ingest_notification_repo)
    complete_upload_use_case = CompleteMultipartUploadUseCase(
        multipart_client=multipart_client,
        notification_repository=ingest_notification_repo,
    )
    orm_session_factory = create_session_factory(cfg.sqlalchemy_dsn)
    session_repository = PostgresSessionRepository(session_factory=orm_session_factory)
    session_clip_repository = PostgresSessionClipRepository(
        session_factory=orm_session_factory
    )
    user_repository = PostgresUserRepository(session_factory=orm_session_factory)
    user_measurements_repository = PostgresUserMeasurementsRepository(
        session_factory=orm_session_factory
    )

    user_id_provider = TokenIdProvider(prefix="usr", length=16)
    login_user_use_case = LoginUserUseCase(
        user_repository=user_repository,
        user_id_provider=user_id_provider,
    )
    save_measurements_use_case = SaveUserMeasurementsUseCase(
        user_repository=user_repository,
        measurements_repository=user_measurements_repository,
    )
    create_session_use_case = CreateSessionUseCase(
        repository=session_repository,
        object_prefix_root=cfg.storage_object_prefix,
        default_max_clips=cfg.session_default_max_clips,
        default_ttl=timedelta(minutes=cfg.session_default_ttl_minutes),
    )
    request_session_clip_upload_use_case = RequestSessionClipUploadUseCase(
        session_repository=session_repository,
        clip_repository=session_clip_repository,
        storage_client=multipart_client,
        url_ttl=timedelta(hours=cfg.upload_expiry_hours),
    )
    clip_event_publisher = RedisClipEventPublisher(
        host=cfg.redis_host,
        port=cfg.redis_port,
        db=cfg.redis_db,
        channel=cfg.redis_channel,
        bucket_name=cfg.storage_bucket,
    )
    session_clip_uploaded_use_case = SessionClipUploadedUseCase(
        session_repository=session_repository,
        clip_repository=session_clip_repository,
        event_publisher=clip_event_publisher,
    )

    app.include_router(
        create_router(
            create_upload_use_case,
            notify_ingest_use_case,
            complete_upload_use_case,
            create_session_use_case,
            request_session_clip_upload_use_case,
            session_clip_uploaded_use_case,
            login_user_use_case,
            save_measurements_use_case,
        )
    )

    # Include status and recommendations routes
    app.include_router(create_status_router())

    return app


app = build_app()
