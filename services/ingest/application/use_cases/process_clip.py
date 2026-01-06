from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

from services.ingest.application.interfaces import (
    ClipProcessor,
    SegmentHeaderCache,
    StorageGateway,
)
from services.ingest.domain.clip import ClipJob, ProcessedClip

_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_CLUSTER_MAGIC = b"\x1f\x43\xb6\x75"
_MAX_HEADER_SCAN_BYTES = 512 * 1024


class ProcessClipUseCase:
    def __init__(
        self,
        *,
        storage: StorageGateway,
        processor: ClipProcessor,
        header_cache: SegmentHeaderCache | None = None,
    ) -> None:
        self._storage = storage
        self._processor = processor
        self._header_cache = header_cache

    def execute(self, job: ClipJob) -> ProcessedClip:
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / (Path(job.object_key).name or "input.bin")

            self._storage.download(
                job.source_bucket, job.object_key, input_path.as_posix()
            )
            normalized_path = self._maybe_patch_live_segment(job, input_path)
            processed_path = self._processor.process(
                source=normalized_path,
                workdir=tmpdir_path,
                metadata=job.metadata,
            )
            self._storage.upload(
                job.target_bucket, job.object_key, processed_path.as_posix()
            )

        return ProcessedClip(
            session_id=job.session_id,
            clip_id=job.clip_id,
            bucket=job.target_bucket,
            object_key=job.object_key,
            metadata=job.metadata,
        )

    def _maybe_patch_live_segment(self, job: ClipJob, path: Path) -> Path:
        metadata = job.metadata or {}
        if not _should_patch_live_segment(metadata):
            return path

        if _has_ebml_header(path):
            if self._header_cache is not None:
                header = _extract_webm_header(path)
                if header:
                    self._header_cache.set(job.session_id, header)
            return path

        if self._header_cache is None:
            return path
        cached_header = self._header_cache.get(job.session_id)
        if not cached_header:
            return path
        patched_path = path.parent / f"patched_{path.name}"
        _prepend_header(cached_header, source=path, destination=patched_path)
        return patched_path


def _should_patch_live_segment(metadata: Mapping[str, object]) -> bool:
    if metadata.get("source") != "live":
        return False
    mime = str(metadata.get("mime_type") or "").lower()
    return "webm" in mime


def _has_ebml_header(path: Path) -> bool:
    with path.open("rb") as file_obj:
        prefix = file_obj.read(len(_EBML_MAGIC))
    return prefix.startswith(_EBML_MAGIC)


def _extract_webm_header(path: Path) -> bytes | None:
    buffer = bytearray()
    with path.open("rb") as file_obj:
        while len(buffer) < _MAX_HEADER_SCAN_BYTES:
            chunk = file_obj.read(4096)
            if not chunk:
                break
            buffer.extend(chunk)
            idx = buffer.find(_CLUSTER_MAGIC)
            if idx != -1:
                return bytes(buffer[:idx])
    return None


def _prepend_header(header: bytes, *, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as dest, source.open("rb") as src:
        dest.write(header)
        while True:
            chunk = src.read(8192)
            if not chunk:
                break
            dest.write(chunk)
