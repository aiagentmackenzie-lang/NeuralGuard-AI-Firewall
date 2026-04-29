"""NeuralGuard database session — async session factory and dependency.

Provides session_factory for creating sessions and a FastAPI-style
get_session generator for dependency injection / manual use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from neuralguard.db.engine import get_engine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)


def session_factory() -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the current engine.

    Must be called after create_engine() has been called at startup.
    Returns a sessionmaker that can be used to create sessions.
    """
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Database engine not initialized — call create_engine() first")
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async generator that yields a session and ensures cleanup.

    Usage:
        async for session in get_session():
            session.add(obj)
            await session.commit()
    """
    factory = session_factory()
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
