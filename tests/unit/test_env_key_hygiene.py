"""F5: NEURALGUARD_* env-key hygiene.

Root cause: NEURALGUARD_SERVER_* was used by every config surface
(.env.example, docker-compose.yml, smoke_test.sh, perf.yml, README,
runbooks — even a main.py error message) while the settings classes read
NEURALGUARD_* (env_prefix NEURALGUARD_ + bare field names), and
pydantic-settings' extra="ignore" made the dead names SILENT. The rename
plus the unknown-key detector kills the failure class: a NEURALGUARD_*
key that maps to no settings field is reported in every environment and
refuses production startup.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuralguard.config.settings import NeuralGuardConfig, known_env_keys, unknown_env_keys
from neuralguard.main import lifespan


class TestKnownKeys:
    def test_renamed_server_keys_are_known(self) -> None:
        """The F5 rename: the real names the ServerSettings fields read."""
        known = known_env_keys()
        for key in (
            "NEURALGUARD_HOST",
            "NEURALGUARD_PORT",
            "NEURALGUARD_WORKERS",
            "NEURALGUARD_MAX_REQUEST_BODY_BYTES",
            "NEURALGUARD_CORS_ORIGINS",
            "NEURALGUARD_ALLOW_CREDENTIALS",
            "NEURALGUARD_ALLOW_INSECURE_HTTP",
        ):
            assert key in known, f"{key} missing from known_env_keys()"

    def test_prefixed_subsettings_are_known(self) -> None:
        known = known_env_keys()
        assert "NEURALGUARD_AUTH_API_KEYS" in known
        assert "NEURALGUARD_AGENT_GUARDIAN_BACKEND" in known
        assert "NEURALGUARD_SCANNER_JUDGE_ENABLED" in known


class TestUnknownKeyDetection:
    def test_bogus_env_var_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEURALGUARD_TOTALLY_BOGUS", "1")
        assert "NEURALGUARD_TOTALLY_BOGUS" in unknown_env_keys()

    def test_dead_server_name_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anyone reintroducing the old names gets caught by the detector."""
        monkeypatch.setenv("NEURALGUARD_SERVER_ALLOW_INSECURE_HTTP", "true")
        assert "NEURALGUARD_SERVER_ALLOW_INSECURE_HTTP" in unknown_env_keys()

    def test_bogus_key_in_dotenv_file_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "NEURALGUARD_DEBUG=true\nNEURALGUARD_BOGUS_FILE_KEY=x\n", encoding="utf-8"
        )
        assert "NEURALGUARD_BOGUS_FILE_KEY" in unknown_env_keys()
        assert "NEURALGUARD_DEBUG" not in unknown_env_keys()

    def test_clean_environment_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)  # no .env file
        for key in list(_neuralguard_env_keys()):
            monkeypatch.delenv(key, raising=False)
        assert unknown_env_keys() == []

    def test_case_insensitive_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("neuralguard_bogus_lowercase", "1")
        assert "NEURALGUARD_BOGUS_LOWERCASE" in unknown_env_keys()


def _neuralguard_env_keys() -> list[str]:
    import os

    return [k for k in os.environ if k.upper().startswith("NEURALGUARD_")]


class TestLifespanGate:
    async def test_production_refuses_unknown_keys(self) -> None:
        """Fail-closed: production startup refuses unknown NEURALGUARD_* keys."""
        config = NeuralGuardConfig(environment="production")
        app = SimpleNamespace(state=SimpleNamespace(config=config))
        # The test environment is hermetic (F17 fixture purges NEURALGUARD_*
        # env vars); inject the bogus key directly.
        import os

        os.environ["NEURALGUARD_BOGUS_LIFESPAN_KEY"] = "1"
        try:
            with pytest.raises(RuntimeError, match="unknown NEURALGUARD_"):
                async with lifespan(app):  # type: ignore[arg-type]
                    pass
        finally:
            os.environ.pop("NEURALGUARD_BOGUS_LIFESPAN_KEY", None)

    async def test_development_warns_but_boots(self) -> None:
        """Dev does not fail on unknown keys (loud warning only)."""
        config = NeuralGuardConfig(environment="development")
        app = SimpleNamespace(state=SimpleNamespace(config=config))
        import os

        os.environ["NEURALGUARD_BOGUS_LIFESPAN_KEY"] = "1"
        try:
            async with lifespan(app):  # type: ignore[arg-type]
                pass  # entering the lifespan must NOT raise on the unknown key
        except RuntimeError as exc:
            pytest.fail(f"development lifespan must not refuse on unknown keys: {exc}")
        finally:
            os.environ.pop("NEURALGUARD_BOGUS_LIFESPAN_KEY", None)


class TestRepoSurfaces:
    """The rename is pinned at the config surfaces so it cannot rot back."""

    @pytest.mark.parametrize(
        "path",
        [
            ".env.example",
            "docker-compose.yml",
            "scripts/smoke_test.sh",
            ".github/workflows/perf.yml",
            "README.md",
            "docs/runbooks/tls_termination.md",
            "src/neuralguard/main.py",
        ],
    )
    def test_no_dead_server_names_on_any_surface(self, path: str) -> None:
        content = Path(path).read_text(encoding="utf-8")
        assert "NEURALGUARD_SERVER_" not in content, (
            f"{path} still references the dead NEURALGUARD_SERVER_* names (F5)"
        )

    def test_compose_uses_renamed_keys_and_host_port_var(self) -> None:
        content = Path("docker-compose.yml").read_text(encoding="utf-8")
        assert "NEURALGUARD_ALLOW_INSECURE_HTTP" in content
        assert "NEURALGUARD_MAX_REQUEST_BODY_BYTES" in content
        # The host-port mapping var must NOT collide with the app bind port.
        assert "${NG_HOST_PORT:-8000}:8000" in content
