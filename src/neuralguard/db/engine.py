"""NeuralGuard database engine — AsyncEngine factory.

Creates a SQLAlchemy AsyncEngine from the audit postgres_url setting.
Supports connection pooling with sensible defaults for a middleware workload.
"""

from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = structlog.get_logger(__name__)

# Module-level engine holder — set once at startup, cleared at shutdown
_engine: AsyncEngine | None = None


def create_engine(postgres_url: str, echo: bool = False) -> AsyncEngine:
    """Create an async SQLAlchemy engine for audit persistence.

    Args:
        postgres_url: PostgreSQL connection string (asyncpg driver).
        echo: Enable SQL echo logging (debug mode).

    Returns:
        Configured AsyncEngine instance.
    """
    global _engine
    _engine = create_async_engine(
        postgres_url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
    )
    logger.info("db_engine_created", url_prefix=postgres_url[:40])
    return _engine


def get_engine() -> AsyncEngine | None:
    """Return the current module-level engine, or None if not initialized."""
    return _engine


async def dispose_engine() -> None:
    """Gracefully dispose of the engine and its connection pool."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        logger.info("db_engine_disposed")
        _engine = None
