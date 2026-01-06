from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def _engine_options():
    return {
        "pool_pre_ping": True,
    }


class Base(DeclarativeBase):
    pass


def create_session_factory(dsn: str):
    engine = create_engine(dsn, **_engine_options())
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
