"""Tenant config registry (Sprint C, C1 / ROADMAP P1-2).

Loads ``tenants/*.yaml`` / ``*.yml`` / ``*.json`` into an in-memory dict
keyed by tenant id, polls the directory mtime on a background task, and
re-parses on change. The request path only does dict lookups — it never
touches the filesystem and never raises on a config miss (fail-open to
the global default).

Thread-safety: the registry holds an immutable dict reference and swaps
it atomically on reload (Python assignment is atomic under the GIL).
Readers (sync + async) grab the reference; the reload task holds an
``asyncio.Lock`` so two reload ticks never race.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

import structlog

from neuralguard.tenants.config import TenantConfig, TenantScannerOverrides

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

_SUPPORTED_SUFFIXES = (".yaml", ".yml", ".json")



class TenantConfigRegistry:
    """In-memory cache of per-tenant override configs.

    Construct with the global ``TenantSettings`` (enabled / default_tenant
    / config_path). When ``enabled`` is False the registry is inert:
    ``get`` returns ``None`` for every tenant, so the rate-limit middleware
    and pipeline fall back to the global config unchanged (backward
    compatibility).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        default_tenant: str,
        config_path: Path,
        reload_interval_seconds: float = 30.0,
    ) -> None:
        self.enabled = enabled
        self.default_tenant = default_tenant
        self.config_path = config_path
        self.reload_interval_seconds = reload_interval_seconds
        self._configs: dict[str, TenantConfig] = {}
        self._last_dir_mtime: float | None = None
        self._reload_lock = asyncio.Lock()
        self._reload_task: asyncio.Task[Any] | None = None

    # ── Construction helpers ────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any, *, quiet: bool = False) -> TenantConfigRegistry:
        """Build a registry from a ``TenantSettings`` instance and load once.

        ``settings`` is typed ``Any`` to avoid an import cycle with
        ``config.settings`` (which itself imports nothing from here, but the
        loose typing keeps the dependency direction one-way). ``quiet=True``
        suppresses the load-time INFO notices (forwarded to :meth:`load`).
        """
        registry = cls(
            enabled=settings.enabled,
            default_tenant=settings.default_tenant,
            config_path=settings.config_path,
            reload_interval_seconds=getattr(settings, "reload_interval_seconds", 30.0),
        )
        registry.load(quiet=quiet)
        return registry

    # ── Loading ─────────────────────────────────────────────────────────

    def load(self, quiet: bool = False) -> None:
        """Parse the tenant config directory synchronously (startup / reload).

        Fail-safe: a missing directory, a parse error in one file, or a
        missing YAML extra all leave the previous ``self._configs`` intact
        and log. The registry is NEVER blanked by a failed reload.

        ``quiet=True`` suppresses the load-time INFO notices (``tenants_loaded``
        / ``tenants_dir_missing``) so a CLI caller can keep its stdout clean for
        machine-readable output. WARNING-level notices (malformed files, id
        mismatches) are NEVER suppressed — those are operational errors.
        """
        if not self.enabled:
            self._configs = {}
            return

        path = self.config_path
        if not path.exists() or not path.is_dir():
            if not quiet:
                logger.info("tenants_dir_missing", path=str(path), msg="all tenants use global default")
            self._configs = {}
            self._last_dir_mtime = None
            return

        new_configs: dict[str, TenantConfig] = {}
        for entry in sorted(path.iterdir()):
            if not entry.is_file() or entry.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            try:
                cfg = self._parse_file(entry)
            except Exception as exc:  # parse / validation / IO error
                logger.warning(
                    "tenant_config_skip",
                    file=entry.name,
                    error=repr(exc),
                    msg="skipping malformed tenant file; existing config retained if present",
                )
                # If we already loaded this tenant before, keep the old one.
                existing = self._configs.get(entry.stem)
                if existing is not None:
                    new_configs[existing.tenant_id] = existing
                continue
            if cfg.tenant_id != entry.stem:
                logger.warning(
                    "tenant_config_id_mismatch",
                    file=entry.name,
                    declared=cfg.tenant_id,
                    expected=entry.stem,
                    msg="tenant_id must equal filename stem; skipping",
                )
                existing = self._configs.get(entry.stem)
                if existing is not None:
                    new_configs[existing.tenant_id] = existing
                continue
            new_configs[cfg.tenant_id] = cfg

        # Atomic swap — readers never see a half-built dict.
        self._configs = new_configs
        try:
            self._last_dir_mtime = path.stat().st_mtime
        except OSError:
            self._last_dir_mtime = None
        if not quiet:
            logger.info("tenants_loaded", count=len(new_configs), path=str(path))

    def _parse_file(self, entry: Path) -> TenantConfig:
        """Parse one tenant file into a validated ``TenantConfig``."""
        suffix = entry.suffix.lower()
        if suffix == ".json":
            data = json.loads(entry.read_text(encoding="utf-8"))
        else:  # .yaml / .yml
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    f"tenant file {entry.name} is YAML but PyYAML is not installed. "
                    "Install neuralguard[tenants] or use a .json tenant file."
                ) from exc
            data = yaml.safe_load(entry.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"tenant file {entry.name} must contain a mapping at top level")
        return TenantConfig.model_validate(data)

    # ── Hot-reload ───────────────────────────────────────────────────────

    def _dir_changed(self) -> bool:
        """True if the tenant dir mtime changed since last load."""
        if not self.config_path.exists() or not self.config_path.is_dir():
            return self._last_dir_mtime is not None
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return False
        return mtime != self._last_dir_mtime

    async def reload_if_changed(self) -> bool:
        """Re-parse the tenant dir if its mtime changed. Returns True if reloaded."""
        if not self.enabled:
            return False
        async with self._reload_lock:
            if not self._dir_changed():
                return False
            # Run the blocking parse in a thread so we don't stall the loop.
            await asyncio.to_thread(self.load)
            return True

    async def _reload_loop(self) -> None:
        """Background task: poll the tenant dir mtime and reload on change."""
        try:
            while True:
                await asyncio.sleep(self.reload_interval_seconds)
                try:
                    await self.reload_if_changed()
                except Exception as exc:  # never let the loop die
                    logger.warning("tenants_reload_error", error=repr(exc))
        except asyncio.CancelledError:
            logger.info("tenants_reload_loop_stopped")

    def start_reload_task(self) -> None:
        """Start the background hot-reload poller (idempotent)."""
        if not self.enabled or self.reload_interval_seconds <= 0:
            return
        if self._reload_task is not None and not self._reload_task.done():
            return
        self._reload_task = asyncio.create_task(self._reload_loop(), name="tenants-reload")

    async def stop_reload_task(self) -> None:
        """Cancel the background poller (shutdown)."""
        task = self._reload_task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._reload_task = None

    # ── Request-path resolution ─────────────────────────────────────────

    def get(self, tenant_id: str) -> TenantConfig | None:
        """Return the override for ``tenant_id`` or ``None`` if none configured.

        ``None`` means "use the global default" — the caller MUST treat None
        as inherit-global, never as deny. This is the fail-open contract.
        """
        if not self.enabled:
            return None
        return self._configs.get(tenant_id)

    def effective_rate_limit(
        self,
        tenant_id: str,
        global_rpm: int,
        global_burst: int,
    ) -> tuple[int, int]:
        """Resolve (rpm, burst) for a tenant, inheriting global on miss/None."""
        cfg = self.get(tenant_id)
        if cfg is None:
            return global_rpm, global_burst
        return cfg.effective_rate_limit(global_rpm, global_burst)

    def effective_scanner_overlay(self, tenant_id: str) -> TenantScannerOverrides | None:
        """Return the scanner overlay for a tenant or None if none configured."""
        if not self.enabled:
            return None
        cfg = self._configs.get(tenant_id)
        if cfg is None:
            return None
        return cfg.scanners

    def list_tenants(self) -> list[TenantConfig]:
        """All configured tenants (sorted by id) — for the read-only API/CLI."""
        return [self._configs[k] for k in sorted(self._configs)]

    def snapshot(self) -> dict[str, TenantConfig]:
        """A shallow copy of the current registry — for diagnostics/tests."""
        return dict(self._configs)
