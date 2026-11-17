"""NG-15: the HITL store posture, readable from the running artifact.

Why this file exists. `hitl_store=redis` is the one setting a multi-replica
deployment must get right, and until now the only way to check it was to read
the manifest and trust it. The startup refusal sees `SERVER_WORKERS`, not
replicas behind a Service, so the application cannot enforce the rule for a
multi-pod deployment — the operator can. An operator can only carry that
obligation if the posture is observable per pod: two requests, one per pod,
each answering *which store am I using* and *is it the same store as the other
pod*.

So `/v1/info` reports, inside `mcp_gateway`:

- ``hitl`` — ``null`` when HITL is disabled, otherwise
- ``store`` — ``"memory"`` | ``"redis"`` (what was configured)
- ``shared_across_workers`` — the structural property, derived from the store
  (a decision raised on one worker is visible to every other worker)
- ``store_id`` — a 12-hex fingerprint of the store *endpoint* (scheme, host,
  port, database) with credentials stripped, so two pods can be compared
  without either of them disclosing a password
- ``ttl_seconds``, ``timeout_seconds``, ``max_pending`` — the bounds a drill
  or an operator needs to reason about a held request

The fingerprint deliberately excludes the URL's credentials: it exists to prove
*pods agree*, not to identify a secret, and a hash of a password-bearing URL is
still a password oracle.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import NeuralGuardConfig
from neuralguard.main import create_app

_HEX12 = re.compile(r"^[0-9a-f]{12}$")


def _config(**mcp_overrides: Any) -> NeuralGuardConfig:
    config = NeuralGuardConfig(environment="development")
    defaults: dict[str, Any] = {
        "enabled": True,
        "upstream_url": "http://fake-mcp-server.local",
        "timeout_seconds": 5.0,
        "server_id": "test-mcp",
    }
    defaults.update(mcp_overrides)
    for key, value in defaults.items():
        setattr(config.mcp, key, value)
    return config


async def _info(config: NeuralGuardConfig) -> dict[str, Any]:
    app = create_app(config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.get("/v1/info")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


class TestHitlPostureIsReported:
    async def test_disabled_hitl_reports_null_rather_than_a_store(self) -> None:
        body = await _info(_config(hitl_enabled=False))
        assert body["mcp_gateway"]["hitl"] is None

    async def test_memory_store_is_reported_as_per_worker(self) -> None:
        body = await _info(_config(hitl_enabled=True, hitl_store="memory"))
        hitl = body["mcp_gateway"]["hitl"]
        assert hitl["store"] == "memory"
        # The whole point of the field: a per-worker store must not read as
        # shared, because that is the defect a two-replica deployment ships.
        assert hitl["shared_across_workers"] is False
        assert hitl["store_id"] is None

    async def test_redis_store_is_reported_as_shared_with_a_fingerprint(self) -> None:
        body = await _info(
            _config(
                hitl_enabled=True,
                hitl_store="redis",
                hitl_redis_url="redis://redis:6379/1",
            )
        )
        hitl = body["mcp_gateway"]["hitl"]
        assert hitl["store"] == "redis"
        assert hitl["shared_across_workers"] is True
        assert _HEX12.match(hitl["store_id"]), hitl["store_id"]

    async def test_the_bounds_a_held_request_depends_on_are_reported(self) -> None:
        body = await _info(
            _config(
                hitl_enabled=True,
                hitl_store="redis",
                hitl_redis_url="redis://redis:6379/1",
                hitl_timeout_seconds=10.0,
                hitl_ttl_seconds=300.0,
                hitl_max_pending=64,
            )
        )
        hitl = body["mcp_gateway"]["hitl"]
        assert hitl["timeout_seconds"] == 10.0
        assert hitl["ttl_seconds"] == 300.0
        assert hitl["max_pending"] == 64


class TestStoreFingerprintComparesPodsWithoutLeakingSecrets:
    async def test_two_pods_with_different_credentials_report_the_same_store(self) -> None:
        # A drilled two-pod deployment usually has the password injected per pod;
        # a fingerprint that moved with the credential would report "two
        # different stores" for the correct configuration.
        pod_a = await _info(
            _config(
                hitl_enabled=True,
                hitl_store="redis",
                hitl_redis_url="redis://:first-password@redis:6379/1",
            )
        )
        pod_b = await _info(
            _config(
                hitl_enabled=True,
                hitl_store="redis",
                hitl_redis_url="redis://:second-password@redis:6379/1",
            )
        )
        assert pod_a["mcp_gateway"]["hitl"]["store_id"] == pod_b["mcp_gateway"]["hitl"]["store_id"]

    async def test_a_different_database_is_a_different_store(self) -> None:
        db1 = await _info(
            _config(hitl_enabled=True, hitl_store="redis", hitl_redis_url="redis://redis:6379/1")
        )
        db2 = await _info(
            _config(hitl_enabled=True, hitl_store="redis", hitl_redis_url="redis://redis:6379/2")
        )
        assert db1["mcp_gateway"]["hitl"]["store_id"] != db2["mcp_gateway"]["hitl"]["store_id"]

    async def test_a_different_endpoint_is_a_different_store(self) -> None:
        one = await _info(
            _config(hitl_enabled=True, hitl_store="redis", hitl_redis_url="redis://redis:6379/1")
        )
        two = await _info(
            _config(
                hitl_enabled=True, hitl_store="redis", hitl_redis_url="redis://other-redis:6379/1"
            )
        )
        assert one["mcp_gateway"]["hitl"]["store_id"] != two["mcp_gateway"]["hitl"]["store_id"]

    async def test_the_fingerprint_is_stable_for_one_endpoint(self) -> None:
        config = _config(
            hitl_enabled=True, hitl_store="redis", hitl_redis_url="redis://redis:6379/1"
        )
        first = await _info(config)
        second = await _info(config)
        assert first["mcp_gateway"]["hitl"]["store_id"] == second["mcp_gateway"]["hitl"]["store_id"]

    async def test_the_response_never_echoes_the_redis_url_or_its_password(self) -> None:
        body = await _info(
            _config(
                hitl_enabled=True,
                hitl_store="redis",
                hitl_redis_url="redis://:drill-secret-password@redis:6379/1",
            )
        )
        serialised = str(body)
        assert "drill-secret-password" not in serialised
        assert "redis://" not in serialised
        assert "6379" not in serialised


class TestTheRestOfTheGatewayBlockIsUnchanged:
    async def test_disabled_gateway_still_reports_no_gateway_block(self) -> None:
        body = await _info(_config(enabled=False))
        assert body["mcp_gateway"] is None

    async def test_enabled_gateway_keeps_its_existing_fields(self) -> None:
        body = await _info(_config(hitl_enabled=True, hitl_store="memory"))
        gateway = body["mcp_gateway"]
        assert gateway["enabled"] is True
        assert gateway["mode"] == "strict"
        assert gateway["headers_required"] in {True, False}
        assert isinstance(gateway["signed_baselines"], bool)


@pytest.mark.parametrize("url,database", [("redis://r:6379/0", "0"), ("redis://r:6379", "0")])
async def test_the_default_database_is_the_same_store_spelled_differently(
    url: str, database: str
) -> None:
    explicit = await _info(
        _config(hitl_enabled=True, hitl_store="redis", hitl_redis_url="redis://r:6379/0")
    )
    implicit = await _info(_config(hitl_enabled=True, hitl_store="redis", hitl_redis_url=url))
    assert (
        explicit["mcp_gateway"]["hitl"]["store_id"] == implicit["mcp_gateway"]["hitl"]["store_id"]
    ), database
