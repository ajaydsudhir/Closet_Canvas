from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


def _load_repo_env() -> None:
    """Load the nearest .env starting from this file upward."""
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        env_file = candidate / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return


_load_repo_env()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise ValueError(f"Environment variable {name} is required")
    return value


def _require_env_int(name: str) -> int:
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


@dataclass(frozen=True)
class CaptureConfig:
    max_initial_parts: int
    asset_id_prefix: str
    asset_id_length: int
    upload_expiry_hours: int
    storage_access_key: str
    storage_bucket: str
    storage_endpoint_url: str
    storage_public_endpoint_url: str
    storage_object_prefix: str
    storage_region: str
    storage_secret_key: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    redis_channel: str
    redis_db: int
    redis_host: str
    redis_port: int
    session_default_max_clips: int
    session_default_ttl_minutes: int

    @property
    def database_dsn(self) -> str:
        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return (
            f"postgresql://{user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sqlalchemy_dsn(self) -> str:
        dsn = self.database_dsn
        if dsn.startswith("postgresql://"):
            return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
        return dsn


def load_config() -> CaptureConfig:
    return CaptureConfig(
        max_initial_parts=_require_env_int("CAPTURE_MAX_INITIAL_PARTS"),
        asset_id_prefix=_require_env("CAPTURE_ASSET_ID_PREFIX"),
        asset_id_length=_require_env_int("CAPTURE_ASSET_ID_LENGTH"),
        upload_expiry_hours=_require_env_int("CAPTURE_UPLOAD_EXPIRY_HOURS"),
        storage_access_key=_require_env("CAPTURE_STORAGE_ACCESS_KEY"),
        storage_bucket=_require_env("CAPTURE_STORAGE_BUCKET"),
        storage_endpoint_url=_require_env("CAPTURE_STORAGE_ENDPOINT_URL"),
        storage_public_endpoint_url=os.getenv(
            "CAPTURE_STORAGE_PUBLIC_ENDPOINT_URL",
            _require_env("CAPTURE_STORAGE_ENDPOINT_URL"),
        ),
        storage_object_prefix=_require_env("CAPTURE_STORAGE_OBJECT_PREFIX"),
        storage_region=_require_env("CAPTURE_STORAGE_REGION"),
        storage_secret_key=_require_env("CAPTURE_STORAGE_SECRET_KEY"),
        db_host=_require_env("CAPTURE_DB_HOST"),
        db_port=_require_env_int("CAPTURE_DB_PORT"),
        db_name=_require_env("CAPTURE_DB_NAME"),
        db_user=_require_env("CAPTURE_DB_USER"),
        db_password=_require_env("CAPTURE_DB_PASSWORD"),
        session_default_ttl_minutes=_require_env_int(
            "CAPTURE_SESSION_DEFAULT_TTL_MINUTES"
        ),
        session_default_max_clips=_require_env_int("CAPTURE_SESSION_DEFAULT_MAX_CLIPS"),
        redis_host=os.getenv("CAPTURE_REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("CAPTURE_REDIS_PORT", "6379")),
        redis_db=int(os.getenv("CAPTURE_REDIS_DB", "0")),
        redis_channel=os.getenv("CAPTURE_REDIS_CHANNEL", "clip_ready"),
    )
