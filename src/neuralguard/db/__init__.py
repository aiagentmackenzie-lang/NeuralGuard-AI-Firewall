"""NeuralGuard database package — PostgreSQL audit persistence.

Imports are lazy to avoid requiring sqlalchemy/asyncpg when the
[db] extra is not installed. Use neuralguard.db.engine, .models, .session
directly when the db backend is configured.
"""

__all__ = ["AuditEventORM", "create_engine", "get_session", "session_factory"]


def __getattr__(name: str):
    """Lazy import — only load DB modules when actually accessed."""
    if name == "create_engine":
        from neuralguard.db.engine import create_engine

        return create_engine
    if name == "AuditEventORM":
        from neuralguard.db.models import AuditEventORM

        return AuditEventORM
    if name in ("get_session", "session_factory"):
        from neuralguard.db.session import get_session as gs
        from neuralguard.db.session import session_factory as sf

        if name == "session_factory":
            return sf
        return gs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
