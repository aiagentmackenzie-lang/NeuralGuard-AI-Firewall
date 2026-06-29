"""Tests for CLI entrypoints."""

from __future__ import annotations

import os

import pytest


class TestCliVersion:
    def test_version_prints_and_exits(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["neuralguard", "version"])
        from neuralguard.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "NeuralGuard v0.1.0" in captured.out


class TestCliCanaryMint:
    def test_canary_mint_disabled_exits_2(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv", ["neuralguard", "canary-mint", "sess-1"]
        )
        from neuralguard.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "disabled" in err.lower()

    def test_canary_mint_enabled_prints_token(self, monkeypatch, capsys, tmp_path):
        env = {
            "NEURALGUARD_CANARY_ENABLED": "true",
            "NEURALGUARD_CANARY_SECRET": "x" * 40,
        }
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(
            "sys.argv", ["neuralguard", "canary-mint", "sess-1"]
        )
        from neuralguard.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        out = capsys.readouterr().out.strip()
        assert out.startswith("NGCANARY-")

    def test_canary_mint_json_count(self, monkeypatch, capsys):
        env = {
            "NEURALGUARD_CANARY_ENABLED": "true",
            "NEURALGUARD_CANARY_SECRET": "x" * 40,
        }
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(
            "sys.argv",
            ["neuralguard", "canary-mint", "sess-1", "--count", "3", "--json"],
        )
        import json

        from neuralguard.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["session_id"] == "sess-1"
        assert len(data["tokens"]) == 3


class TestCliTenants:
    def _env(self, tmp_path, enabled=True):
        return {
            "NEURALGUARD_TENANT_ENABLED": "true" if enabled else "false",
            "NEURALGUARD_TENANT_CONFIG_PATH": str(tmp_path),
        }

    def test_tenants_disabled_exits_2(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(os, "environ", self._env(tmp_path, enabled=False))
        monkeypatch.setattr("sys.argv", ["neuralguard", "tenants", "list"])
        from neuralguard.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "disabled" in capsys.readouterr().err.lower()

    def test_tenants_list(self, monkeypatch, capsys, tmp_path):
        import json

        (tmp_path / "acme.json").write_text(
            json.dumps({"tenant_id": "acme", "description": "Acme"}), encoding="utf-8"
        )
        monkeypatch.setattr(os, "environ", self._env(tmp_path))
        monkeypatch.setattr("sys.argv", ["neuralguard", "tenants", "list", "--json"])
        from neuralguard.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)
        assert data[0]["tenant_id"] == "acme"

    def test_tenants_info_unknown_tenant_note(self, monkeypatch, capsys, tmp_path):
        import json

        monkeypatch.setattr(os, "environ", self._env(tmp_path))
        monkeypatch.setattr("sys.argv", ["neuralguard", "tenants", "info", "ghost", "--json"])
        from neuralguard.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["configured"] is False
        assert data["tenant_id"] == "ghost"


class TestCliServe:
    def test_serve_no_args_runs(self, monkeypatch, capsys):
        called = {}

        def fake_serve_main():
            called["yes"] = True

        monkeypatch.setattr("neuralguard.cli.serve_main", fake_serve_main)
        monkeypatch.setattr("sys.argv", ["neuralguard", "serve"])
        from neuralguard.cli import main

        main()
        assert called.get("yes")

    def test_serve_with_overrides(self, monkeypatch):
        fake_environ: dict[str, str] = {}
        monkeypatch.setattr(os, "environ", fake_environ)
        monkeypatch.setattr(
            "sys.argv", ["neuralguard", "serve", "--host", "127.0.0.1", "--port", "9000"]
        )
        monkeypatch.setattr("neuralguard.cli.serve_main", lambda: None)
        from neuralguard.cli import main

        main()
        assert fake_environ.get("NEURALGUARD_HOST") == "127.0.0.1"
        assert fake_environ.get("NEURALGUARD_PORT") == "9000"
