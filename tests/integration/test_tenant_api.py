"""Integration tests for per-tenant config API + lifespan gates (Sprint C, C1)."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import (
    AuthSettings,
    NeuralGuardConfig,
    ServerSettings,
    TenantSettings,
)
from neuralguard.main import create_app


def _write_tenant(path, body: dict) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def tenants_dir(tmp_path):
    d = tmp_path / "tenants"
    d.mkdir()
    _write_tenant(
        d / "acme.json",
        {
            "tenant_id": "acme",
            "description": "Acme Co",
            "requests_per_minute": 100,
            "burst_size": 20,
            "scanners": {"semantic": False},
        },
    )
    _write_tenant(
        d / "globex.json",
        {"tenant_id": "globex", "requests_per_minute": 50},
    )
    return d


def _app(tenants_dir, *, enabled: bool = True, auth: AuthSettings | None = None) -> object:
    cfg = NeuralGuardConfig(environment="development")
    cfg.tenant = TenantSettings(enabled=enabled, config_path=tenants_dir)
    if auth is not None:
        cfg.auth = auth
    return create_app(cfg)


@pytest.fixture
async def tenants_client(tenants_dir):
    app = _app(tenants_dir)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    # Run the lifespan so the registry + reload task are live.
    async with (
        AsyncClient(transport=transport, base_url="http://test") as c,
        app.router.lifespan_context(app),  # type: ignore[attr-defined]
    ):
        yield c


# ── GET /v1/tenants ────────────────────────────────────────────────────────


class TestListTenants:
    @pytest.mark.asyncio
    async def test_list_returns_all_configured(self, tenants_client: AsyncClient):
        r = await tenants_client.get("/v1/tenants")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        ids = {t["tenant_id"] for t in data["tenants"]}
        assert ids == {"acme", "globex"}

    @pytest.mark.asyncio
    async def test_list_includes_effective_resolution(self, tenants_client: AsyncClient):
        r = await tenants_client.get("/v1/tenants")
        data = r.json()
        acme = next(t for t in data["tenants"] if t["tenant_id"] == "acme")
        assert acme["configured"] is True
        assert acme["requests_per_minute"] == 100
        assert acme["effective_requests_per_minute"] == 100
        assert acme["scanners"]["semantic"] is False
        # structural + pattern always True; semantic narrowed off by tenant.
        assert acme["effective_scanners"]["structural"] is True
        assert acme["effective_scanners"]["pattern"] is True
        assert acme["effective_scanners"]["semantic"] is False

    @pytest.mark.asyncio
    async def test_list_404_when_disabled(self, tenants_dir):
        app = _app(tenants_dir, enabled=False)
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/v1/tenants")
            assert r.status_code == 404
            assert r.json()["detail"]["error"] == "tenants_disabled"


# ── GET /v1/tenants/{tenant_id} ─────────────────────────────────────────────


class TestGetTenant:
    @pytest.mark.asyncio
    async def test_get_configured_tenant(self, tenants_client: AsyncClient):
        r = await tenants_client.get("/v1/tenants/acme")
        assert r.status_code == 200
        data = r.json()
        assert data["tenant_id"] == "acme"
        assert data["configured"] is True

    @pytest.mark.asyncio
    async def test_get_unknown_tenant_fail_open(self, tenants_client: AsyncClient):
        # A config miss is NOT a 404 — fail-open to the global defaults.
        r = await tenants_client.get("/v1/tenants/ghost")
        assert r.status_code == 200
        data = r.json()
        assert data["configured"] is False
        assert data["tenant_id"] == "ghost"
        # Global defaults apply.
        assert data["effective_requests_per_minute"] > 0

    @pytest.mark.asyncio
    async def test_get_404_when_disabled(self, tenants_dir):
        app = _app(tenants_dir, enabled=False)
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/v1/tenants/acme")
            assert r.status_code == 404
            assert r.json()["detail"]["error"] == "tenants_disabled"

    @pytest.mark.asyncio
    async def test_secrets_not_leaked(self, tenants_client: AsyncClient):
        r = await tenants_client.get("/v1/tenants/acme")
        # TenantConfig carries no secrets; assert the response has no secret keys.
        assert "secret" not in r.text.lower()


# ── Tenant binding enforcement ─────────────────────────────────────────────


class TestTenantBinding:
    @pytest.mark.asyncio
    async def test_key_bound_tenant_cannot_read_other_tenant(self, tenants_dir):
        app = _app(
            tenants_dir,
            auth=AuthSettings(enabled=True, api_keys=["key|acme"]),
        )
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with (
            AsyncClient(transport=transport, base_url="http://test") as c,
            app.router.lifespan_context(app),  # type: ignore[attr-defined]
        ):
            # Authenticated as acme -> reading globex is forbidden.
                r = await c.get(
                    "/v1/tenants/globex", headers={"X-API-Key": "key"}
                )
                assert r.status_code == 403
                assert r.json()["detail"]["error"] == "tenant_mismatch"
                # Reading own tenant is allowed.
                r2 = await c.get(
                    "/v1/tenants/acme", headers={"X-API-Key": "key"}
                )
                assert r2.status_code == 200


# ── Lifespan production gate (YAML without PyYAML) ──────────────────────────


class TestTenantLifespanGates:
    async def test_production_yaml_without_pyyaml_refused(self, tmp_path, monkeypatch):
        d = tmp_path / "tenants"
        d.mkdir()
        (d / "acme.yaml").write_text("tenant_id: acme\n", encoding="utf-8")

        # Simulate PyYAML being unavailable for the lifespan check.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        cfg = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=1),
        )
        cfg.tenant = TenantSettings(enabled=True, config_path=d)
        app = create_app(cfg)
        with pytest.raises(RuntimeError, match=r"YAML tenant files but PyYAML"):
            async with app.router.lifespan_context(app):
                pass

    async def test_production_json_without_pyyaml_allowed(self, tmp_path):
        d = tmp_path / "tenants"
        d.mkdir()
        (d / "acme.json").write_text(
            json.dumps({"tenant_id": "acme"}), encoding="utf-8"
        )
        cfg = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=1),
        )
        cfg.tenant = TenantSettings(enabled=True, config_path=d)
        app = create_app(cfg)
        async with app.router.lifespan_context(app):
            pass  # should not raise — JSON needs no extra
        assert app.state.tenant_registry is not None
        assert len(app.state.tenant_registry.list_tenants()) == 1

    def test_disabled_tenant_mode_no_registry_on_state(self):
        app = create_app()
        assert app.state.tenant_registry is None

    async def test_enabled_tenant_mode_registry_built_and_reload_task_started(
        self, tenants_dir
    ):
        app = _app(tenants_dir)
        async with app.router.lifespan_context(app):  # type: ignore[attr-defined]
            assert app.state.tenant_registry is not None
            assert app.state.tenant_registry.enabled is True
            assert app.state.tenant_registry._reload_task is not None
        # After shutdown the reload task is cancelled.
        assert app.state.tenant_registry._reload_task is None
