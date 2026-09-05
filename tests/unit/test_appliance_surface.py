"""Structural pins for the standalone appliance surface (compose-boot drill, 2026-09-05).

The drill caught a real ship-blocker: the Docker image was built without the
``redis`` extra while the appliance compose enables redis-backed Agent
Guardian + rate limiting — a guaranteed ModuleNotFoundError crash loop at
boot, invisible to the test suite (which never builds the image).

These pins make the Docker/compose agreement explicit so the image can never
silently rot away from the appliance posture again.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


class TestDockerfileExtras:
    """The image must carry every backend the appliance compose can enable."""

    def test_sync_includes_redis_extra(self) -> None:
        # F22: uv sync without --extra redis + AG/ratelimit backend=redis
        # = ModuleNotFoundError at boot (crash loop). Caught live by the drill.
        sync_line = next(line for line in _read("Dockerfile").splitlines() if "uv sync" in line)
        assert "--extra redis" in sync_line, (
            "Dockerfile uv sync must include --extra redis: the appliance compose "
            "sets AG + ratelimit backends to redis"
        )

    def test_sync_includes_db_extra(self) -> None:
        # postgres audit backend (AUDIT_BACKEND=postgres) needs asyncpg/sqlalchemy.
        sync_line = next(line for line in _read("Dockerfile").splitlines() if "uv sync" in line)
        assert "--extra db" in sync_line

    def test_sync_includes_tenants_extra(self) -> None:
        # NEURALGUARD_TENANT_* / YAML tenant registry is a documented appliance
        # knob — without pyyaml the knob is a boot failure.
        sync_line = next(line for line in _read("Dockerfile").splitlines() if "uv sync" in line)
        assert "--extra tenants" in sync_line

    def test_sync_includes_metrics_extra(self) -> None:
        # /v1/metrics (Prometheus) is part of the appliance posture.
        sync_line = next(line for line in _read("Dockerfile").splitlines() if "uv sync" in line)
        assert "--extra metrics" in sync_line


class TestApplianceComposePosture:
    """docker-compose.appliance.yml must keep the fail-fast + redis posture."""

    def test_required_vars_fail_fast(self) -> None:
        compose = _read("docker-compose.appliance.yml")
        for var in (
            "NEURALGUARD_AUTH_API_KEYS",
            "NEURALGUARD_PROXY_UPSTREAM_URL",
            "NEURALGUARD_CANARY_SECRET",
        ):
            assert f"${{{var}:?" in compose, (
                f"{var} must use the :? fail-fast interpolation so a misconfigured "
                "appliance refuses to start instead of booting unauthenticated"
            )

    def test_stateful_backends_use_redis(self) -> None:
        compose = _read("docker-compose.appliance.yml")
        assert "NEURALGUARD_AGENT_GUARDIAN_BACKEND=redis" in compose
        assert "NEURALGUARD_RATELIMIT_BACKEND=redis" in compose

    def test_proxy_enabled_and_auth_enabled(self) -> None:
        compose = _read("docker-compose.appliance.yml")
        assert "NEURALGUARD_PROXY_ENABLED=true" in compose
        assert "NEURALGUARD_AUTH_ENABLED=true" in compose

    def test_no_plaintext_port_binding_on_redis_or_postgres(self) -> None:
        # Neither redis nor postgres publishes to the host — they are internal
        # to the compose network only.
        compose = _read("docker-compose.appliance.yml")
        redis_svc = compose.split("  redis:")[1]
        assert "ports:" not in redis_svc.split("healthcheck:")[0], (
            "redis must not publish host ports in the appliance profile"
        )
        postgres_svc = compose.split("  postgres:")[1]
        assert "ports:" not in postgres_svc.split("volumes:")[0], (
            "postgres must not publish host ports in the appliance profile"
        )
