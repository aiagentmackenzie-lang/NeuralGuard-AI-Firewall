"""NeuralGuard — LLM Guard / AI Application Firewall.

FastAPI application factory and server entrypoint.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuralguard.api.routes import router
from neuralguard.config.settings import NeuralGuardConfig, load_config
from neuralguard.logging.audit import AuditLogger
from neuralguard.middleware.ratelimit import RateLimitMiddleware
from neuralguard.scanners.pattern import PatternScanner
from neuralguard.scanners.pipeline import ScannerPipeline
from neuralguard.scanners.structural import StructuralScanner

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING default
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown logic.

    Replaces deprecated @app.on_event("startup"/"shutdown").
    """
    config: NeuralGuardConfig = app.state.config

    # ── Startup ──
    structlog.get_logger("neuralguard").info(
        "startup",
        version=config.version,
        environment=config.environment,
        host=config.server.host,
        port=config.server.port,
    )

    # Initialize PostgreSQL engine if audit backend is postgres
    if config.audit.backend == "postgres" and config.audit.postgres_url:
        try:
            from neuralguard.db.engine import create_engine as db_create_engine
            from neuralguard.db.models import Base

            engine = db_create_engine(config.audit.postgres_url)
            # Create tables if they don't exist (dev/staging convenience)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            structlog.get_logger("neuralguard").info("db_tables_created", backend="postgres")
        except ImportError:
            structlog.get_logger("neuralguard").warning(
                "db_deps_missing",
                msg="asyncpg/sqlalchemy not installed; JSONL fallback active",
            )
        except Exception as exc:
            structlog.get_logger("neuralguard").error(
                "db_init_failed", error=str(exc), msg="PostgreSQL init failed; JSONL fallback"
            )

    yield

    # ── Shutdown ──
    if config.audit.backend == "postgres" and config.audit.postgres_url:
        try:
            from neuralguard.db.engine import dispose_engine

            await dispose_engine()
        except Exception:
            pass  # Best-effort cleanup

    structlog.get_logger("neuralguard").info("shutdown")


def create_app(config: NeuralGuardConfig | None = None) -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description="LLM Guard / AI Application Firewall — defensive middleware for prompt injection, jailbreak detection, and agentic security",
        docs_url="/docs" if config.environment != "production" else None,
        redoc_url="/redoc" if config.environment != "production" else None,
        lifespan=lifespan,
    )

    # ── Store config and services on app state ──
    app.state.config = config
    app.state.start_time = time.time()

    # ── Initialize scanner pipeline ──
    pipeline = ScannerPipeline(config)
    pipeline.register_scanner(StructuralScanner(config.scanner))
    pipeline.register_scanner(PatternScanner(config.scanner))
    app.state.pipeline = pipeline

    # ── Initialize audit logger ──
    audit_logger = AuditLogger(config.audit)
    app.state.audit_logger = audit_logger

    # ── Middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, settings=config.rate_limit)

    # ── Routes ──
    app.include_router(router)

    return app


def main() -> None:
    """CLI entrypoint."""
    import uvicorn

    config = load_config()

    # Configure structlog level from config
    import logging

    log_level = getattr(logging, config.server.log_level)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    uvicorn.run(
        "neuralguard.main:create_app",
        factory=True,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level=config.server.log_level.lower(),
    )


if __name__ == "__main__":
    main()
