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
