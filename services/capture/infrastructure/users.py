"""User repository implementation using PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Float

from domain.user import User, UserMeasurements
from infrastructure.db import Base


class UserRecord(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=False)


class UserMeasurementsRecord(Base):
    __tablename__ = "user_measurements"

    user_id = Column(String, primary_key=True)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    chest = Column(Float, nullable=True)
    waist = Column(Float, nullable=True)
    hips = Column(Float, nullable=True)
    shoulder_width = Column(Float, nullable=True)
    inseam = Column(Float, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class PostgresUserRepository:
    """PostgreSQL implementation of UserRepository."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def create(self, user: User) -> User:
        """Create a new user."""
        record = UserRecord(
            user_id=user.user_id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )
        with self._session_factory() as db:
            db.add(record)
            db.commit()
        return user

    def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        with self._session_factory() as db:
            record = (
                db.query(UserRecord).filter(UserRecord.email == email).one_or_none()
            )
            if record is None:
                return None
            return User(
                user_id=record.user_id,
                email=record.email,
                name=record.name,
                created_at=record.created_at,
                last_login_at=record.last_login_at,
            )

    def get_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        with self._session_factory() as db:
            record = db.get(UserRecord, user_id)
            if record is None:
                return None
            return User(
                user_id=record.user_id,
                email=record.email,
                name=record.name,
                created_at=record.created_at,
                last_login_at=record.last_login_at,
            )

    def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        with self._session_factory() as db:
            record = db.get(UserRecord, user_id)
            if record:
                record.last_login_at = datetime.now(timezone.utc)
                db.commit()


class PostgresUserMeasurementsRepository:
    """PostgreSQL implementation of UserMeasurementsRepository."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def save(self, measurements: UserMeasurements) -> UserMeasurements:
        """Save or update user measurements."""
        with self._session_factory() as db:
            record = db.get(UserMeasurementsRecord, measurements.user_id)
            now = datetime.now(timezone.utc)

            if record is None:
                # Create new record
                record = UserMeasurementsRecord(
                    user_id=measurements.user_id,
                    height=measurements.height,
                    weight=measurements.weight,
                    chest=measurements.chest,
                    waist=measurements.waist,
                    hips=measurements.hips,
                    shoulder_width=measurements.shoulder_width,
                    inseam=measurements.inseam,
                    updated_at=now,
                )
                db.add(record)
            else:
                # Update existing record
                if measurements.height is not None:
                    record.height = measurements.height
                if measurements.weight is not None:
                    record.weight = measurements.weight
                if measurements.chest is not None:
                    record.chest = measurements.chest
                if measurements.waist is not None:
                    record.waist = measurements.waist
                if measurements.hips is not None:
                    record.hips = measurements.hips
                if measurements.shoulder_width is not None:
                    record.shoulder_width = measurements.shoulder_width
                if measurements.inseam is not None:
                    record.inseam = measurements.inseam
                record.updated_at = now

            db.commit()
            db.refresh(record)

            return UserMeasurements(
                user_id=record.user_id,
                height=record.height,
                weight=record.weight,
                chest=record.chest,
                waist=record.waist,
                hips=record.hips,
                shoulder_width=record.shoulder_width,
                inseam=record.inseam,
                updated_at=record.updated_at,
            )

    def get(self, user_id: str) -> UserMeasurements | None:
        """Get user measurements by user ID."""
        with self._session_factory() as db:
            record = db.get(UserMeasurementsRecord, user_id)
            if record is None:
                return None
            return UserMeasurements(
                user_id=record.user_id,
                height=record.height,
                weight=record.weight,
                chest=record.chest,
                waist=record.waist,
                hips=record.hips,
                shoulder_width=record.shoulder_width,
                inseam=record.inseam,
                updated_at=record.updated_at,
            )
