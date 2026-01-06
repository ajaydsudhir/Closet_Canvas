"""Save user measurements use case."""

from __future__ import annotations

from datetime import datetime, timezone

from application.user_interfaces import UserMeasurementsRepository, UserRepository
from domain.user import UserMeasurements


class SaveUserMeasurementsUseCase:
    """Use case for saving user body measurements."""

    def __init__(
        self,
        user_repository: UserRepository,
        measurements_repository: UserMeasurementsRepository,
    ):
        self._user_repository = user_repository
        self._measurements_repository = measurements_repository

    def execute(
        self,
        user_id: str,
        height: float | None = None,
        weight: float | None = None,
        chest: float | None = None,
        waist: float | None = None,
        hips: float | None = None,
        shoulder_width: float | None = None,
        inseam: float | None = None,
    ) -> UserMeasurements:
        """
        Save or update user measurements.

        Args:
            user_id: User ID
            height: Height in cm
            weight: Weight in kg
            chest: Chest circumference in cm
            waist: Waist circumference in cm
            hips: Hips circumference in cm
            shoulder_width: Shoulder width in cm
            inseam: Inseam length in cm

        Returns:
            UserMeasurements entity

        Raises:
            ValueError: If user doesn't exist
        """
        # Verify user exists
        user = self._user_repository.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        measurements = UserMeasurements(
            user_id=user_id,
            height=height,
            weight=weight,
            chest=chest,
            waist=waist,
            hips=hips,
            shoulder_width=shoulder_width,
            inseam=inseam,
            updated_at=datetime.now(timezone.utc),
        )

        return self._measurements_repository.save(measurements)
