from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol


class StorageGateway(Protocol):
    def download(self, bucket: str, object_key: str, destination_path: str) -> None: ...

    def upload(self, bucket: str, object_key: str, source_path: str) -> None: ...


class ClipProcessor(Protocol):
    def process(
        self,
        *,
        source: Path,
        workdir: Path,
        metadata: Mapping[str, object] | None = None,
    ) -> Path: ...


class SegmentHeaderCache(Protocol):
    def set(self, session_id: str, header: bytes) -> None: ...

    def get(self, session_id: str) -> bytes | None: ...


class VideoGatingService(Protocol):
    """Evaluates video quality and generates segmentation masks."""

    def evaluate(self, video_path: Path) -> tuple[bool, list]:
        """Returns (allowed, masks) tuple."""
        ...
