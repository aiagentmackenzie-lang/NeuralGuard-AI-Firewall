"""Tests for the tenant config registry (Sprint C, C1 / P1-2)."""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from neuralguard.config.settings import TenantSettings
from neuralguard.tenants.config import TenantConfig, TenantScannerOverrides
from neuralguard.tenants.registry import TenantConfigRegistry


def _write(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestRegistryLoad:
    def test_disabled_registry_is_inert(self, tmp_path):
        reg = TenantConfigRegistry(
            enabled=False,
            default_tenant="default",
            config_path=tmp_path,
        )
        reg.load()
        assert reg.get("acme") is None
        assert reg.effective_rate_limit("acme", 60, 10) == (60, 10)
        assert reg.effective_scanner_overlay("acme") is None
        assert reg.list_tenants() == []

    def test_missing_dir_loads_empty(self, tmp_path):
        reg = TenantConfigRegistry(
            enabled=True,
            default_tenant="default",
            config_path=tmp_path / "nope",
        )
        reg.load()
        assert reg.list_tenants() == []
        # Unknown tenant -> None -> caller uses global default (fail-open).
        assert reg.get("acme") is None
        assert reg.effective_rate_limit("acme", 60, 10) == (60, 10)

    def test_empty_dir_loads_empty(self, tmp_path):
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        assert reg.list_tenants() == []

    def test_loads_yaml_and_json(self, tmp_path):
        _write(
            tmp_path / "acme.yaml",
            "tenant_id: acme\nrequests_per_minute: 100\nscanners:\n  semantic: false\n",
        )
        _write(
            tmp_path / "globex.json",
            json.dumps(
                {
                    "tenant_id": "globex",
                    "requests_per_minute": 200,
                    "burst_size": 20,
                }
            ),
        )
        # An unrelated file type is ignored.
        _write(tmp_path / "README.md", "# not a tenant")
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        ids = [t.tenant_id for t in reg.list_tenants()]
        assert ids == ["acme", "globex"]
        assert reg.get("acme").requests_per_minute == 100
        assert reg.get("acme").scanners.semantic is False
        assert reg.get("globex").requests_per_minute == 200

    def test_tenant_id_must_match_filename_stem(self, tmp_path):
        # acme.yaml declares tenant_id: globex -> mismatch, skipped (last-good
        # retained if a prior load had it; here no prior, so dropped).
        _write(tmp_path / "acme.yaml", "tenant_id: globex\n")
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        assert reg.list_tenants() == []

    def test_malformed_file_kept_from_previous_load(self, tmp_path):
        _write(tmp_path / "acme.yaml", "tenant_id: acme\nrequests_per_minute: 100\n")
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        assert reg.get("acme").requests_per_minute == 100
        # Corrupt the file — the previous good config is retained.
        _write(tmp_path / "acme.yaml", "tenant_id: acme\n  bad: : :")
        reg.load()
        assert reg.get("acme").requests_per_minute == 100

    def test_unknown_tenant_returns_none_not_raising(self, tmp_path):
        _write(tmp_path / "acme.yaml", "tenant_id: acme\n")
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        # The fail-open contract: missing tenant config never raises.
        assert reg.get("does-not-exist") is None
        assert reg.effective_rate_limit("does-not-exist", 60, 10) == (60, 10)

    def test_yaml_without_pyyaml_raises_parse_error(self, tmp_path, monkeypatch):
        _write(tmp_path / "acme.yaml", "tenant_id: acme\n")
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        # Simulate PyYAML being unavailable.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reg.load()
        # YAML file skipped due to missing PyYAML; JSON would still work.
        assert reg.list_tenants() == []

    def test_from_settings_loads(self, tmp_path):
        _write(tmp_path / "acme.json", json.dumps({"tenant_id": "acme"}))
        settings = TenantSettings(enabled=True, config_path=tmp_path)
        reg = TenantConfigRegistry.from_settings(settings)
        assert [t.tenant_id for t in reg.list_tenants()] == ["acme"]


class TestRegistryHotReload:
    def test_dir_changed_detection(self, tmp_path):
        _write(tmp_path / "acme.yaml", "tenant_id: acme\nrequests_per_minute: 100\n")
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        assert not reg._dir_changed()
        # Bump mtime by writing a new file.
        time.sleep(0.01)
        _write(tmp_path / "globex.yaml", "tenant_id: globex\nrequests_per_minute: 50\n")
        os.utime(tmp_path, None)
        assert reg._dir_changed()

    @pytest.mark.asyncio
    async def test_reload_if_changed_picks_up_new_tenant(self, tmp_path):
        _write(tmp_path / "acme.yaml", "tenant_id: acme\nrequests_per_minute: 100\n")
        reg = TenantConfigRegistry(
            enabled=True,
            default_tenant="default",
            config_path=tmp_path,
            reload_interval_seconds=0.01,
        )
        reg.load()
        assert reg.get("globex") is None
        # Add a new tenant file and bump dir mtime.
        time.sleep(0.01)
        _write(tmp_path / "globex.yaml", "tenant_id: globex\nrequests_per_minute: 50\n")
        os.utime(tmp_path, None)
        reloaded = await reg.reload_if_changed()
        assert reloaded is True
        assert reg.get("globex").requests_per_minute == 50
        # Second call with no change -> no reload.
        reloaded = await reg.reload_if_changed()
        assert reloaded is False

    @pytest.mark.asyncio
    async def test_reload_task_start_stop(self, tmp_path):
        reg = TenantConfigRegistry(
            enabled=True,
            default_tenant="default",
            config_path=tmp_path,
            reload_interval_seconds=0.05,
        )
        reg.load()
        reg.start_reload_task()
        assert reg._reload_task is not None
        await asyncio.sleep(0.08)
        await reg.stop_reload_task()
        assert reg._reload_task is None

    def test_reload_task_disabled_when_interval_zero(self, tmp_path):
        reg = TenantConfigRegistry(
            enabled=True,
            default_tenant="default",
            config_path=tmp_path,
            reload_interval_seconds=0,
        )
        reg.load()
        reg.start_reload_task()
        assert reg._reload_task is None


class TestRegistryResolution:
    def test_effective_scanner_overlay_inherits_none(self, tmp_path):
        _write(
            tmp_path / "acme.yaml",
            "tenant_id: acme\nscanners:\n  agent_guardian: null\n  semantic: false\n",
        )
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        overlay = reg.effective_scanner_overlay("acme")
        assert overlay.agent_guardian is None
        assert overlay.semantic is False
        # Unknown tenant -> None (inherit global).
        assert reg.effective_scanner_overlay("ghost") is None

    def test_snapshot_is_a_copy(self, tmp_path):
        _write(tmp_path / "acme.yaml", "tenant_id: acme\n")
        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        snap = reg.snapshot()
        assert "acme" in snap
        snap["extra"] = TenantConfig(tenant_id="extra")  # type: ignore[assignment]
        # Mutating the snapshot must not affect the registry.
        assert "extra" not in reg.snapshot()
