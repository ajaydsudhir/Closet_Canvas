from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping

from services.ingest.application.interfaces import ClipProcessor
from services.ingest.config import IngestConfig

FASTSTART_EXTENSIONS = {".mp4", ".m4v", ".mov"}


class FFmpegRemuxError(RuntimeError):
    """Raised when ffmpeg fails to remux a clip."""


class FFmpegClipProcessor(ClipProcessor):
    def __init__(
        self,
        *,
        target_extension: str = ".mp4",
        live_source_only: bool = True,
        log_level: str = "error",
    ) -> None:
        self._target_extension = (
            target_extension
            if target_extension.startswith(".")
            else f".{target_extension}"
        )
        self._live_source_only = live_source_only
        self._log_level = log_level

    def process(
        self,
        *,
        source: Path,
        workdir: Path,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        metadata = metadata or {}
        if self._live_source_only and metadata.get("source") != "live":
            return source

        destination = self._build_destination_path(workdir)
        self._run_ffmpeg(source, destination)
        return destination

    def _build_destination_path(self, workdir: Path) -> Path:
        return workdir / f"remuxed{self._target_extension}"

    def _run_ffmpeg(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            self._log_level,
            "-i",
            source.as_posix(),
            "-c",
            "copy",
        ]
        if destination.suffix.lower() in FASTSTART_EXTENSIONS:
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(destination.as_posix())
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise FFmpegRemuxError(
                f"ffmpeg remux failed for {source.name}: {stderr.strip() or 'unknown error'}"
            )


class PassthroughClipProcessor(ClipProcessor):
    """Processor that leaves clips untouched."""

    def process(
        self,
        *,
        source: Path,
        workdir: Path,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        return source


def create_clip_processor(config: IngestConfig) -> ClipProcessor:
    return FFmpegClipProcessor(target_extension=config.remux_target_extension)
