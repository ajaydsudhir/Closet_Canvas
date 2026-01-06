"""User domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class User:
    """User entity."""

    user_id: str
    email: str
    name: str
    created_at: datetime
    last_login_at: datetime


@dataclass(frozen=True)
class UserMeasurements:
    """User body measurements."""

    user_id: str
    height: float | None = None
    weight: float | None = None
    chest: float | None = None
    waist: float | None = None
    hips: float | None = None
    shoulder_width: float | None = None
    inseam: float | None = None
    updated_at: datetime | None = None
