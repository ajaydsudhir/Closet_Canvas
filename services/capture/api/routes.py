from __future__ import annotations

from typing import Any, Dict, List, Literal
from datetime import datetime

from fastapi import APIRouter, status, HTTPException
from pydantic import BaseModel, Field

from application.create_multipart_upload import CreateMultipartUploadUseCase
from application.complete_multipart_upload import CompleteMultipartUploadUseCase
from application.create_session import CreateSessionUseCase
from application.request_session_clip_upload import RequestSessionClipUploadUseCase
from application.session_clip_uploaded import SessionClipUploadedUseCase
from application.dto import (
    CompleteMultipartUploadCommand,
    CompletionPart,
    CreateMultipartUploadCommand,
    CreateSessionCommand,
    IngestNotificationCommand,
    RequestSessionClipUploadCommand,
    SessionClipUploadedCommand,
)
from application.notify_ingest import NotifyIngestUseCase
from domain.ingest_notification import IngestNotification
from domain.upload import MultipartUpload, CompletedMultipartUpload
from domain.session import CaptureSession


class MultipartUploadPartResponse(BaseModel):
    part_no: int
    url: str


class MultipartUploadResponse(BaseModel):
    asset_id: str
    upload_id: str
    object_key: str
    parts: List[MultipartUploadPartResponse]
    expires_at: str

    @classmethod
    def from_domain(cls, upload: MultipartUpload) -> "MultipartUploadResponse":
        return cls(
            asset_id=upload.asset_id,
            upload_id=upload.upload_id,
            object_key=upload.object_key,
            parts=[
                MultipartUploadPartResponse(part_no=part.part_no, url=part.url)
                for part in upload.parts
            ],
            expires_at=upload.expires_at.isoformat().replace("+00:00", "Z"),
        )


class MultipartUploadRequest(BaseModel):
    filename: str
    content_type: str
    part_size_bytes: int
    max_parts: int


class IngestNotificationRequest(BaseModel):
    asset_id: str
    upload_id: str
    part_no: int
    etag: str
    client_meta: Dict[str, Any] = Field(default_factory=dict)


class IngestNotificationResponse(BaseModel):
    asset_id: str
    upload_id: str
    part_no: int
    etag: str
    client_meta: Dict[str, Any]
    notified_at: str

    @classmethod
    def from_domain(
        cls, notification: IngestNotification
    ) -> "IngestNotificationResponse":
        return cls(
            asset_id=notification.asset_id,
            upload_id=notification.upload_id,
            part_no=notification.part_no,
            etag=notification.etag,
            client_meta=dict(notification.client_meta),
            notified_at=notification.notified_at.isoformat().replace("+00:00", "Z"),
        )


class MultipartUploadCompletionPart(BaseModel):
    part_no: int
    etag: str


class MultipartUploadCompletionRequest(BaseModel):
    asset_id: str
    upload_id: str
    object_key: str
    parts: List[MultipartUploadCompletionPart]


class MultipartUploadCompletionResponse(BaseModel):
    asset_id: str
    upload_id: str
    object_key: str
    location: str | None

    @classmethod
    def from_domain(
        cls, completed: CompletedMultipartUpload
    ) -> "MultipartUploadCompletionResponse":
        return cls(
            asset_id=completed.asset_id,
            upload_id=completed.upload_id,
            object_key=completed.object_key,
            location=completed.location,
        )


class CreateSessionRequest(BaseModel):
    max_clips: int | None = Field(default=None, ge=1)
    ttl_minutes: int | None = Field(default=None, ge=1, le=480)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    session_id: str
    token: str
    status: str
    object_prefix: str
    max_clips: int
    created_at: str
    expires_at: str

    @classmethod
    def from_domain(cls, session: CaptureSession) -> "CreateSessionResponse":
        return cls(
            session_id=session.session_id,
            token=session.token,
            status=session.status,
            object_prefix=session.object_prefix,
            max_clips=session.max_clips,
            created_at=session.created_at.isoformat().replace("+00:00", "Z"),
            expires_at=session.expires_at.isoformat().replace("+00:00", "Z"),
        )


class CreateSessionClipRequest(BaseModel):
    filename: str
    content_type: str


class CreateSessionClipResponse(BaseModel):
    clip_id: str
    object_key: str
    upload_url: str
    expires_at: str


class ClipMetadataPayload(BaseModel):
    source: Literal["manual", "live"]
    mime_type: str | None = None
    sequence_no: int | None = Field(default=None, ge=0)
    capture_started_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class SessionClipUploadedRequest(BaseModel):
    metadata: ClipMetadataPayload


class UserLoginRequest(BaseModel):
    email: str


class UserLoginResponse(BaseModel):
    id: str
    email: str
    name: str


class UserMeasurementsRequest(BaseModel):
    height: float
    weight: float
    chest: float | None = None
    waist: float | None = None
    hips: float | None = None


def create_router(
    create_upload_use_case: CreateMultipartUploadUseCase,
    notify_ingest_use_case: NotifyIngestUseCase,
    complete_upload_use_case: CompleteMultipartUploadUseCase,
    create_session_use_case: CreateSessionUseCase,
    request_session_clip_upload_use_case: RequestSessionClipUploadUseCase,
    session_clip_uploaded_use_case: SessionClipUploadedUseCase,
    login_user_use_case,
    save_measurements_use_case,
) -> APIRouter:
    router = APIRouter()
    uploads_router = APIRouter(prefix="/v1/uploads", tags=["uploads"])
    sessions_router = APIRouter(prefix="/v1/sessions", tags=["sessions"])
    users_router = APIRouter(prefix="/v1/users", tags=["users"])

    @uploads_router.post(
        "/multipart", response_model=MultipartUploadResponse, status_code=201
    )
    async def create_multipart_upload_endpoint(payload: MultipartUploadRequest):
        command = CreateMultipartUploadCommand(
            filename=payload.filename,
            content_type=payload.content_type,
            part_size_bytes=payload.part_size_bytes,
            max_parts=payload.max_parts,
        )
        result = create_upload_use_case.execute(command)
        return MultipartUploadResponse.from_domain(result)

    @uploads_router.post(
        "/multipart/complete",
        response_model=MultipartUploadCompletionResponse,
        status_code=200,
    )
    async def complete_multipart_upload_endpoint(
        payload: MultipartUploadCompletionRequest,
    ):
        command = CompleteMultipartUploadCommand(
            asset_id=payload.asset_id,
            upload_id=payload.upload_id,
            object_key=payload.object_key,
            parts=[
                CompletionPart(part_no=part.part_no, etag=part.etag)
                for part in payload.parts
            ],
        )
        completed = complete_upload_use_case.execute(command)
        return MultipartUploadCompletionResponse.from_domain(completed)

    ingest_router = APIRouter(prefix="/v1/ingest", tags=["ingest"])

    @sessions_router.post("", response_model=CreateSessionResponse, status_code=201)
    async def create_session_endpoint(payload: CreateSessionRequest):
        command = CreateSessionCommand(
            max_clips=payload.max_clips,
            ttl_minutes=payload.ttl_minutes,
            metadata=payload.metadata,
        )
        session = create_session_use_case.execute(command)
        return CreateSessionResponse.from_domain(session)

    @sessions_router.post(
        "/{session_id}/clips",
        response_model=CreateSessionClipResponse,
        status_code=201,
    )
    async def create_session_clip_endpoint(
        session_id: str, payload: CreateSessionClipRequest
    ):
        command = RequestSessionClipUploadCommand(
            session_id=session_id,
            filename=payload.filename,
            content_type=payload.content_type,
        )
        try:
            result = request_session_clip_upload_use_case.execute(command)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CreateSessionClipResponse(**result)

    @sessions_router.post(
        "/{session_id}/clips/{clip_id}/uploaded",
        status_code=202,
    )
    async def session_clip_uploaded_endpoint(
        session_id: str,
        clip_id: str,
        payload: SessionClipUploadedRequest,
    ):
        command = SessionClipUploadedCommand(
            session_id=session_id,
            clip_id=clip_id,
            metadata=payload.metadata.model_dump(mode="json"),
        )
        try:
            session_clip_uploaded_use_case.execute(command)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "accepted"}

    @ingest_router.post(
        "/notify",
        response_model=IngestNotificationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def notify_ingest_endpoint(payload: IngestNotificationRequest):
        command = IngestNotificationCommand(
            asset_id=payload.asset_id,
            upload_id=payload.upload_id,
            part_no=payload.part_no,
            etag=payload.etag,
            client_meta=payload.client_meta,
        )
        notification = notify_ingest_use_case.execute(command)
        return IngestNotificationResponse.from_domain(notification)

    # User routes
    @users_router.post("/login", response_model=UserLoginResponse, status_code=200)
    async def login_user_endpoint(payload: UserLoginRequest):
        """User login endpoint - creates user if doesn't exist, updates last login if exists."""
        user = login_user_use_case.execute(payload.email)

        return UserLoginResponse(id=user.user_id, email=user.email, name=user.name)

    @users_router.post("/{user_id}/measurements", status_code=204)
    async def submit_measurements(user_id: str, payload: UserMeasurementsRequest):
        """Store user body measurements."""
        try:
            save_measurements_use_case.execute(
                user_id=user_id,
                height=payload.height,
                weight=payload.weight,
                chest=payload.chest,
                waist=payload.waist,
                hips=payload.hips,
            )
            return None
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    router.include_router(uploads_router)
    router.include_router(ingest_router)
    router.include_router(sessions_router)
    router.include_router(users_router)

    return router
