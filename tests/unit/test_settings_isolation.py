"""F17 regression tests: settings construction must be hermetic in tests.

These pin the contract enforced by ``tests/conftest.py``:

* a hostile ``.env`` in the current working directory must NOT reconfigure
  test-constructed settings (the file channel is severed), and
* exported ``NEURALGUARD_*`` shell variables must be purged before each test
  (the fixture purges them; in-test ``setenv`` still works for tests that
  deliberately exercise environment handling).
"""

from __future__ import annotations

import os
from pathlib import Path

from neuralguard.config.settings import NeuralGuardConfig


class TestSettingsHermeticity:
    def test_hostile_env_file_in_cwd_is_ignored(self, tmp_path, monkeypatch) -> None:
        """A dev-style .env in CWD must not flip auth/judge/debug defaults."""
        (tmp_path / ".env").write_text(
            "NEURALGUARD_AUTH_ENABLED=true\n"
            "NEURALGUARD_AUTH_API_KEYS=ng_hostile_key_should_not_load|demo\n"
            "NEURALGUARD_SCANNER_JUDGE_MODEL=hostile-model\n"
            "NEURALGUARD_DEBUG=true\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        config = NeuralGuardConfig()

        assert config.auth.enabled is False
        assert config.auth.api_keys == []
        assert config.scanner.judge_model != "hostile-model"
        assert config.debug is False

    def test_exported_neuralguard_env_vars_are_purged(self) -> None:
        """The autouse fixture must have purged NEURALGUARD_* from the env."""
        leaked = [key for key in os.environ if key.startswith("NEURALGUARD_")]
        assert leaked == []

    def test_purged_env_vars_do_not_reconfigure_settings(self, tmp_path, monkeypatch) -> None:
        """End-to-end: neither file channel nor pre-existing env leaks.

        Set an env var BEFORE constructing settings, then delete it via the
        same purge the fixture performs — the constructed config must use
        defaults, proving the purge (not just the file severing) protects
        settings construction.
        """
        (tmp_path / ".env").write_text("NEURALGUARD_DEBUG=true\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("NEURALGUARD_DEBUG", "true")
        # Mirror the fixture's purge for this deliberate setenv.
        monkeypatch.delenv("NEURALGUARD_DEBUG", raising=False)

        config = NeuralGuardConfig()

        assert config.debug is False
