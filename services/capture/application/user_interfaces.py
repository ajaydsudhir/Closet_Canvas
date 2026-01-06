"""Interfaces for user repositories."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from domain.user import User, UserMeasurements


class UserRepository(Protocol):
    """Repository for user persistence."""

    @abstractmethod
    def create(self, user: User) -> User:
        """Create a new user."""
        ...

    @abstractmethod
    def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        ...

    @abstractmethod
    def get_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        ...

    @abstractmethod
    def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        ...


class UserMeasurementsRepository(Protocol):
    """Repository for user measurements persistence."""

    @abstractmethod
    def save(self, measurements: UserMeasurements) -> UserMeasurements:
        """Save or update user measurements."""
        ...

    @abstractmethod
    def get(self, user_id: str) -> UserMeasurements | None:
        """Get user measurements by user ID."""
        ...
