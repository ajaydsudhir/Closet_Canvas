"""Login user use case."""

from __future__ import annotations

from datetime import datetime, timezone

from application.user_interfaces import UserRepository
from domain.user import User
from infrastructure.ids import TokenIdProvider


class LoginUserUseCase:
    """Use case for user login - creates user if doesn't exist, updates last login if exists."""

    def __init__(
        self,
        user_repository: UserRepository,
        user_id_provider: TokenIdProvider,
    ):
        self._user_repository = user_repository
        self._user_id_provider = user_id_provider

    def execute(self, email: str) -> User:
        """
        Login user by email. Creates new user if doesn't exist, otherwise updates last login.

        Args:
            email: User's email address

        Returns:
            User entity
        """
        # Check if user exists
        existing_user = self._user_repository.get_by_email(email)

        if existing_user:
            # Update last login
            self._user_repository.update_last_login(existing_user.user_id)
            # Return updated user
            return User(
                user_id=existing_user.user_id,
                email=existing_user.email,
                name=existing_user.name,
                created_at=existing_user.created_at,
                last_login_at=datetime.now(timezone.utc),
            )

        # Create new user
        user_id = self._user_id_provider.generate()
        name = email.split("@")[0].title()  # Simple name extraction
        now = datetime.now(timezone.utc)

        new_user = User(
            user_id=user_id,
            email=email,
            name=name,
            created_at=now,
            last_login_at=now,
        )

        return self._user_repository.create(new_user)
