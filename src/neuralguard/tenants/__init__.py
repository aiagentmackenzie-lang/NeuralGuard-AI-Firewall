"""Per-tenant configuration (Sprint C, C1 / ROADMAP P1-2).

Loads ``tenants/<id>.yaml`` (or ``.json``) override files into an
in-memory registry keyed by tenant id, and resolves the effective
per-tenant rate-limit and scanner-set configuration at request time.

Design constraints (security-first):

- **Structural + Pattern are mandatory.** A tenant can never disable the
  core sanitization + regex layers — those are the baseline defense. Only
  the three opt-in scanners (Agent Guardian, Semantic, Judge) are
  overridable per tenant.
- **Override = None means "inherit the global default."** Every override
  field defaults to ``None`` so an empty / partial tenant file degrades to
  the global config, never to an unsafe zero.
- **Unknown tenant -> default tenant config -> global.** A config miss is
  fail-OPEN to the global default, never a 403 — denying a request because
  a YAML file is missing is a self-inflicted denial-of-service.
- **The registry is the hot-path cache.** YAML/JSON is parsed at load +
  reload time only; request resolution is a dict lookup. The request path
  never touches the filesystem.
- **Hot-reload is fail-safe.** A parse error keeps the last-good config and
  logs; it never raises into the request path and never blanks the
  registry.
- **Tenant config is a ceiling for the client ``scanners`` request field.**
  A client may narrow the scanner set but never widen it past what the
  tenant allows (and what is globally registered). This is the per-tenant
  enforcement point.

See ``TenantConfig`` and ``TenantConfigRegistry`` below.
"""

from neuralguard.tenants.config import TenantConfig, TenantScannerOverrides
from neuralguard.tenants.registry import TenantConfigRegistry

__all__ = ["TenantConfig", "TenantConfigRegistry", "TenantScannerOverrides"]
