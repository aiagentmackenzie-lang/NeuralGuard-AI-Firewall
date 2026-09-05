"""NeuralGuard — LLM Guard / AI Application Firewall.

FastAPI application factory and server entrypoint.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neuralguard.api.routes import router
from neuralguard.config.settings import NeuralGuardConfig, load_config, unknown_env_keys
from neuralguard.logging.audit import AuditLogger
from neuralguard.metrics import metrics
from neuralguard.middleware.auth import AuthMiddleware
from neuralguard.middleware.bodysize import BodySizeMiddleware
from neuralguard.middleware.ratelimit import RateLimitMiddleware
from neuralguard.models.schemas import ScanLayer
from neuralguard.net.egress import is_private_endpoint
from neuralguard.scanners.pattern import PatternScanner
from neuralguard.scanners.pipeline import ScannerPipeline
from neuralguard.scanners.structural import StructuralScanner

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def _build_processors(environment: str) -> list[Any]:
    """Return structlog processors — JSON in production, console in dev."""
    if environment == "production":
        return [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]


structlog.configure(
    processors=_build_processors("development"),
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING default
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


def _is_private_judge_url(url: str) -> bool:
    """True if the judge endpoint is loopback/private (F10.3 egress gate).

    Private = loopback literals, RFC1918, link-local, IPv6 loopback/ULA,
    dot-less hostnames (container-internal names like ``ollama``), and the
    Docker host reference ``host.docker.internal``. Everything else (public
    IPs and dot-ful public hostnames) is EGRESS.
    """
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host == "host.docker.internal":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal: bare names are container-internal; public
        # hostnames have dots and are egress.
        return "." not in host
    return addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_unspecified


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

    # F10.5 warmup: load the judge model once at startup so the circuit
    # breaker does not trip on cold-start timeouts of the first judged
    # request. Runs in the lifespan (not create_app — tests construct apps
    # without making model calls). Non-fatal by design.
    judge_scanner = getattr(app.state, "judge_scanner", None)
    if judge_scanner is not None:
        app.state.judge_warmup_ok = judge_scanner.warmup()

    # Unknown NEURALGUARD_* env keys (F5): a typo'd or stale key is a silent
    # no-op today. Production refuses to start; dev/staging logs a loud
    # warning with the offending keys.
    unknown = unknown_env_keys()
    if unknown:
        if config.environment == "production":
            raise RuntimeError(
                "Production startup refused: unknown NEURALGUARD_* environment "
                f"keys {unknown} — they map to no settings field and would "
                "silently do nothing. Fix the names (see .env.example) or "
                "remove them."
            )
        structlog.get_logger("neuralguard").warning(
            "unknown_neuralguard_env_keys",
            keys=unknown,
            msg="These NEURALGUARD_* keys map to no settings field and silently "
            "do nothing (typo or stale name?). Production would refuse to start.",
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

    # ── Per-tenant registry hot-reload (Sprint C, C1) ──
    # The registry was constructed in create_app; start its background poller
    # now that the event loop is running. Best-effort: a failure to start the
    # poller is non-fatal (the registry still serves its initial load).
    tenant_registry = getattr(app.state, "tenant_registry", None)
    if tenant_registry is not None:
        try:
            tenant_registry.start_reload_task()
            structlog.get_logger("neuralguard").info(
                "tenants_reload_started",
                interval=tenant_registry.reload_interval_seconds,
            )
        except RuntimeError as exc:
            structlog.get_logger("neuralguard").warning(
                "tenants_reload_start_failed", error=str(exc)
            )

    # ── Production security fail-fast ──
    # Refuse to serve in production unless authentication is configured with at
    # least one API key. This prevents accidental open deployments.
    if config.environment == "production":
        if not config.auth.enabled:
            raise RuntimeError(
                "Production startup refused: authentication is disabled. "
                "Set NEURALGUARD_AUTH_ENABLED=true and NEURALGUARD_AUTH_API_KEYS."
            )
        if not config.auth.api_keys:
            raise RuntimeError(
                "Production startup refused: no API keys configured. Set NEURALGUARD_AUTH_API_KEYS."
            )
        # Multi-worker rate limiting: the in-memory limiter is per-process, so
        # with workers > 1 a tenant gets (limit + burst) * workers requests per
        # window. Refuse to start unless the Redis backend is configured.
        if (
            config.rate_limit.enabled
            and config.server.workers > 1
            and config.rate_limit.backend != "redis"
        ):
            raise RuntimeError(
                f"Production startup refused: rate_limit.backend={config.rate_limit.backend} "
                f"with server.workers={config.server.workers} allows a tenant to exceed the "
                "rate limit by a factor of the worker count. Set "
                "NEURALGUARD_RATELIMIT_BACKEND=redis and NEURALGUARD_RATELIMIT_REDIS_URL."
            )
        if (
            config.rate_limit.enabled
            and config.rate_limit.backend == "redis"
            and not config.rate_limit.redis_url
        ):
            raise RuntimeError(
                "Production startup refused: rate_limit.backend=redis but no redis_url set. "
                "Set NEURALGUARD_RATELIMIT_REDIS_URL."
            )
        # Agent Guardian backend sanity (F4). The redis backend is implemented
        # (shared signal store); still refuse a config that asks for it
        # without a URL rather than failing at first request.
        if (
            config.agent_guardian.enabled
            and config.agent_guardian.backend == "redis"
            and not config.agent_guardian.redis_url
        ):
            raise RuntimeError(
                "Production startup refused: agent_guardian.backend=redis but no "
                "redis_url set. Set NEURALGUARD_AGENT_GUARDIAN_REDIS_URL, or use "
                "backend=memory (single-worker only)."
            )
        if (
            config.agent_guardian.enabled
            and config.agent_guardian.backend == "memory"
            and config.server.workers > 1
        ):
            structlog.get_logger("neuralguard").warning(
                "agent_guardian_memory_multi_worker",
                msg="agent_guardian.backend=memory with workers>1: each worker keeps "
                "its own session window, so a session split across workers is not "
                "correlated (degraded multi-turn detection). Use backend=redis for "
                "multi-worker production.",
            )
        # Canary token verification (Phase 3, Sprint B, B3). Refuse to serve in
        # production if the feature is enabled but the secret is missing or too
        # short — a weak/empty secret makes the canary trivially guessable, so
        # the exfiltration signal is worthless.
        if config.canary.enabled:
            if not config.canary.secret:
                raise RuntimeError(
                    "Production startup refused: canary.enabled=true but "
                    "NEURALGUARD_CANARY_SECRET is not set. Set a strong secret "
                    "(>= 32 chars), or disable canary."
                )
            if len(config.canary.secret) < 32:
                raise RuntimeError(
                    "Production startup refused: canary secret is shorter than "
                    "32 characters — the canary would be guessable. Set a stronger "
                    "NEURALGUARD_CANARY_SECRET, or disable canary."
                )
        # Per-tenant config (Sprint C, C1). If tenant mode is on and a YAML
        # tenant file is present, PyYAML must be installed — otherwise the
        # registry silently skips every YAML tenant (silent partial failure
        # in production is worse than a loud refusal). Operators can either
        # `pip install neuralguard[tenants]` or use .json tenant files.
        if config.tenant.enabled:
            tenant_path = config.tenant.config_path
            if tenant_path.exists() and tenant_path.is_dir():
                has_yaml = any(
                    p.suffix.lower() in (".yaml", ".yml")
                    for p in tenant_path.iterdir()
                    if p.is_file()
                )
                if has_yaml:
                    try:
                        import yaml  # noqa: F401
                    except ImportError as exc:
                        raise RuntimeError(
                            "Production startup refused: tenant mode is enabled with "
                            "YAML tenant files but PyYAML is not installed. Install "
                            "neuralguard[tenants], or use .json tenant files."
                        ) from exc
        # TLS enforcement: refuse plain HTTP unless explicitly allowed (behind a
        # TLS-terminating reverse proxy that the operator takes responsibility for).
        if not config.server.allow_insecure_http:
            structlog.get_logger("neuralguard").warning(
                "production_tls_notice",
                msg="Production mode active. Terminate TLS at a reverse proxy (nginx/Caddy/Traefik) "
                "or set --ssl-keyfile/--ssl-certfile. Set NEURALGUARD_ALLOW_INSECURE_HTTP=true "
                "ONLY if a TLS-terminating proxy is in front.",
            )
        else:
            structlog.get_logger("neuralguard").warning(
                "production_insecure_http_allowed",
                msg="allow_insecure_http=true: ensure a TLS-terminating reverse proxy is in front.",
            )

    # Judge egress gate (F10.3): production ENFORCES a loopback/private judge
    # endpoint unless the operator opts in explicitly (logged + surfaced in
    # readiness). Cloud-via-Ollama models route via the local daemon but
    # inference happens at the vendor — that IS egress.
    if (
        config.environment == "production"
        and config.scanner.judge_enabled
        and not _is_private_judge_url(config.scanner.judge_ollama_url)
    ):
        if config.scanner.judge_allow_egress:
            structlog.get_logger("neuralguard").warning(
                "judge_egress_explicitly_allowed",
                url=config.scanner.judge_ollama_url,
                msg="Judge prompts LEAVE the trust boundary (explicit opt-in). "
                "Never enable for sensitive workloads.",
            )
        else:
            raise RuntimeError(
                "Production startup refused: judge_ollama_url "
                f"({config.scanner.judge_ollama_url}) is not loopback/private. "
                "Prompts would leave the trust boundary. Set "
                "NEURALGUARD_SCANNER_JUDGE_ALLOW_EGRESS=true to opt in "
                "explicitly (logged, and surfaced in /v1/ready)."
            )

    yield

    # ── Shutdown ──
    # Stop the per-tenant hot-reload poller first (Sprint C, C1).
    tenant_registry = getattr(app.state, "tenant_registry", None)
    if tenant_registry is not None:
        await tenant_registry.stop_reload_task()

    if config.audit.backend == "postgres" and config.audit.postgres_url:
        try:
            from neuralguard.db.engine import dispose_engine

            await dispose_engine()
        except Exception:
            pass  # Best-effort cleanup

    # Close the Redis rate-limiter connection if one was created.
    redis_limiter = getattr(app.state, "redis_limiter", None)
    if redis_limiter is not None:
        await redis_limiter.aclose()

    # Close the Agent Guardian redis session-store connection (F4).
    ag_store = getattr(app.state, "agent_guardian_redis_store", None)
    if ag_store is not None:
        import contextlib

        with contextlib.suppress(Exception):
            ag_store.aclose()

    # Close the proxy forwarder's HTTP client (F9).
    proxy_forwarder = getattr(app.state, "proxy_forwarder", None)
    if proxy_forwarder is not None:
        import contextlib

        with contextlib.suppress(Exception):
            await proxy_forwarder.aclose()

    structlog.get_logger("neuralguard").info("shutdown")


def create_app(config: NeuralGuardConfig | None = None) -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    if config is None:
        config = load_config()

    # Configure structured logging for this environment (JSON in production).
    structlog.configure(
        processors=_build_processors(config.environment),
        wrapper_class=structlog.make_filtering_bound_logger(
            _log_level_int(config.server.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

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
    app.state.metrics = metrics

    # ── Initialize scanner pipeline ──
    pipeline = ScannerPipeline(config)
    pipeline.register_scanner(StructuralScanner(config.scanner))
    pipeline.register_scanner(PatternScanner(config.scanner))

    # Register the Agent Guardian scanner if enabled (Phase 3, Sprint B).
    if config.agent_guardian.enabled:
        from neuralguard.scanners.agent_guardian import AgentGuardianScanner

        ag_scanner = AgentGuardianScanner(config.agent_guardian)
        pipeline.register_scanner(ag_scanner)
        # F4: stash the redis-backed session store so the lifespan can close
        # the connection cleanly on shutdown (same pattern as the limiter).
        if ag_scanner._redis_store is not None:
            app.state.agent_guardian_redis_store = ag_scanner._redis_store
        structlog.get_logger("neuralguard").info(
            "agent_guardian_registered",
            backend=config.agent_guardian.backend,
            window_turns=config.agent_guardian.session_window_turns,
        )

    # Register semantic scanner if enabled and dependencies available
    if config.scanner.semantic_enabled:
        try:
            from neuralguard.semantic.similarity import SimilarityScanner

            pipeline.register_scanner(SimilarityScanner(config.scanner))
            structlog.get_logger("neuralguard").info("semantic_scanner_registered")
        except (ImportError, FileNotFoundError) as exc:
            structlog.get_logger("neuralguard").warning(
                "semantic_scanner_unavailable",
                error=str(exc),
                msg="Install neuralguard[semantic] and run export/corpus scripts",
            )

    # Register LLM-as-Judge scanner if enabled
    if config.scanner.judge_enabled:
        try:
            from neuralguard.semantic.judge import JudgeScanner

            judge_scanner = JudgeScanner(config.scanner)
            pipeline.register_scanner(judge_scanner)
            structlog.get_logger("neuralguard").info(
                "judge_scanner_registered", model=config.scanner.judge_model
            )
        except (ImportError, FileNotFoundError) as exc:
            structlog.get_logger("neuralguard").warning(
                "judge_scanner_unavailable",
                error=str(exc),
                msg="Ensure Ollama is running with the configured model",
            )

    app.state.pipeline = pipeline

    # F10.5: stash the judge scanner for the lifespan warmup (the warmup runs
    # at STARTUP — constructing apps in tests must not make model calls).
    app.state.judge_scanner = pipeline._scanners.get(ScanLayer.JUDGE)

    # ── Initialize the per-tenant config registry (Sprint C, C1) ──
    # Only constructed when enabled; its absence on app.state means multi-tenant
    # mode is off and the rate-limit middleware + pipeline fall back to global
    # config (backward compatible). The registry loads tenants/*.yaml|json once
    # here; the lifespan starts the background hot-reload poller.
    if config.tenant.enabled:
        from neuralguard.tenants import TenantConfigRegistry

        tenant_registry = TenantConfigRegistry.from_settings(config.tenant)
        app.state.tenant_registry = tenant_registry
        pipeline.set_tenant_registry(tenant_registry)
        structlog.get_logger("neuralguard").info(
            "tenant_registry_registered",
            tenants=len(tenant_registry.list_tenants()),
            reload_interval=tenant_registry.reload_interval_seconds,
        )
    else:
        app.state.tenant_registry = None

    # ── Initialize the canary token manager (Phase 3, B3) ──
    # Only constructed when enabled; its absence on app.state means the feature
    # is off (routes check for None). The secret is not logged here.
    if config.canary.enabled:
        from neuralguard.canary import CanaryManager

        app.state.canary_manager = CanaryManager(config.canary)
        structlog.get_logger("neuralguard").info(
            "canary_manager_registered",
            token_count=config.canary.token_count,
        )
    else:
        app.state.canary_manager = None

    # ── Initialize audit logger ──
    audit_logger = AuditLogger(config.audit)
    app.state.audit_logger = audit_logger

    # ── Standalone appliance proxy (F9) ──
    # OFF by default; enabled = NeuralGuard becomes a transparent guardian.
    app.state.proxy_forwarder = None
    if config.proxy.enabled:
        if not config.proxy.upstream_url.strip():
            raise RuntimeError(
                "proxy.enabled=true but upstream_url is empty: the proxy would "
                "have nowhere to forward. Set NEURALGUARD_PROXY_UPSTREAM_URL."
            )
        from neuralguard.proxy.forwarder import UpstreamForwarder

        app.state.proxy_forwarder = UpstreamForwarder(config.proxy)
        egress = (
            "local"
            if is_private_endpoint(config.proxy.upstream_url)
            else "cloud (prompts LEAVE the trust boundary)"
        )
        structlog.get_logger("neuralguard").info(
            "proxy_enabled",
            upstream_egress=egress,
            timeout_seconds=config.proxy.timeout_seconds,
            msg="Standalone appliance proxy ENABLED. "
            f"Upstream egress: {egress}. The upstream API key is held server-side.",
        )

    # ── Middleware ──
    # Order matters: outermost first. Body-size limit must run BEFORE the app
    # parses JSON, so it is added first (outermost). Auth runs before rate
    # limiting so the rate limiter can key on the authenticated tenant.
    #
    # The Redis rate-limiter client (if backend=redis) is constructed here and
    # stored on app.state so the lifespan can close it cleanly on shutdown.
    redis_limiter: Any = None
    redis_client_for_mw: Any = None
    if config.rate_limit.enabled and config.rate_limit.backend == "redis":
        from neuralguard.middleware.ratelimit_redis import RedisRateLimiter

        redis_limiter = RedisRateLimiter(config.rate_limit)
        redis_client_for_mw = redis_limiter.client
        app.state.redis_limiter = redis_limiter

    cors_origins = config.server.cors_origins
    # In production, if cors_origins is empty, use a restrictive default (no CORS)
    if config.environment == "production" and not cors_origins:
        cors_origins = []  # No CORS — API-only, no browser access
    # Safety: never combine allow_credentials=true with a wildcard origin.
    if config.server.allow_credentials and ("*" in cors_origins):
        structlog.get_logger("neuralguard").warning(
            "cors_misconfiguration",
            msg="allow_credentials=True with wildcard origin is invalid; forcing allow_credentials=False.",
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=config.server.allow_credentials and ("*" not in cors_origins),
        allow_methods=["POST", "GET"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        settings=config.rate_limit,
        redis_client=redis_client_for_mw,
        tenant_registry=app.state.tenant_registry,
    )
    app.add_middleware(AuthMiddleware, settings=config.auth)
    app.add_middleware(BodySizeMiddleware, max_bytes=config.server.max_request_body_bytes)

    # ── Routes ──
    app.include_router(router)
    # /v1/info (service metadata, extended with F9/F10.3 egress posture) is
    # in the main router. The proxy routes mount only when enabled (F9).
    if config.proxy.enabled:
        from neuralguard.api.routes_proxy import router as proxy_router

        app.include_router(proxy_router)

    # ── Global exception handler: request_id correlation + sanitized 500 ──
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        corr_id = str(uuid.uuid4())
        structlog.get_logger("neuralguard").error(
            "unhandled_exception",
            correlation_id=corr_id,
            path=request.url.path,
            error=repr(exc),
        )
        # Avoid leaking internals to the client.
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An internal error occurred.",
                "correlation_id": corr_id,
            },
        )

    return app


def _log_level_int(level: str) -> int:
    import logging

    return getattr(logging, level, logging.INFO)


def main() -> None:
    """CLI entrypoint."""
    import uvicorn

    config = load_config()

    # Configure structlog level from config
    import logging

    log_level = getattr(logging, config.server.log_level)
    structlog.configure(
        processors=_build_processors(config.environment),
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
