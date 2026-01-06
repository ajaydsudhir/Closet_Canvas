from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from services.ingest.application.interfaces import (
    StorageGateway,
    VideoGatingService,
)
from services.ingest.domain.clip import GatingJob, GatingResult


class GateClipUseCase:
    """
    Use case for gating video clips through quality validation.
    Follows clean architecture with dependency injection of services.
    """

    def __init__(
        self,
        *,
        storage: StorageGateway,
        gating_service: VideoGatingService,
    ) -> None:
        self._storage = storage
        self._gating_service = gating_service

    def execute(self, job: GatingJob) -> GatingResult:
        """
        Execute the gating workflow:
        1. Download video from storage
        2. Run quality validation and get masks
        3. Return gating result with masks
        """
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            video_path = tmpdir_path / (Path(job.object_key).name or "video.mp4")

            # Download video
            self._storage.download(
                job.source_bucket, job.object_key, video_path.as_posix()
            )

            # Run gating evaluation
            allowed, masks = self._gating_service.evaluate(video_path)

            return GatingResult(
                session_id=job.session_id,
                clip_id=job.clip_id,
                passed=allowed and len(masks) > 0,
                masks=masks if allowed else [],
                metadata=job.metadata,
            )
