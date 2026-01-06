from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ClipJob:
    session_id: str
    clip_id: str
    source_bucket: str
    object_key: str
    target_bucket: str
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ProcessedClip:
    session_id: str
    clip_id: str
    bucket: str
    object_key: str
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class GatingJob:
    session_id: str
    clip_id: str
    source_bucket: str
    object_key: str
    frames_bucket: str
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class GatingResult:
    session_id: str
    clip_id: str
    passed: bool
    masks: list
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SMPLJob:
    session_id: str
    clip_id: str
    source_bucket: str
    object_key: str
    masks_bucket: str
    masks_keys: list[str]
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SMPLResult:
    session_id: str
    clip_id: str
    success: bool
    output_bucket: str
    output_key: str
    metadata: Mapping[str, object] | None = None
