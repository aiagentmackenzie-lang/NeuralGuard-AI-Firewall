"""Fleet deployment compose (GAP-3 closure) — offline posture pins.

The fleet file wires NeuralGuard + SecurityScarletAI into ONE compose
project. These pins are text-based (no YAML dep, no sibling checkout
needed) so they run in CI:

- every NEURALGUARD_* env key the fleet file sets maps to a real settings
  field (the F5 unknown-key REFUSE gate would otherwise kill the boot in
  production);
- NeuralGuard does NOT publish :8000 (Scarlet owns it — the whole reason
  the fleet port exists);
- the SIEM + MCP + judge wiring is present and pointed at the right
  compose-network service names;
- the fleet-critical tokens fail-fast (:? interpolation).
"""

from __future__ import annotations

import re
from pathlib import Path

from neuralguard.config.settings import known_env_keys

_FLEET_COMPOSE = Path("deploy/fleet/docker-compose.fleet.yml")
_FLEET_ENV_EXAMPLE = Path("deploy/fleet/fleet.env.example")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestFleetComposePosture:
    def test_every_neuralguard_env_key_is_known_to_f5_gate(self) -> None:
        """Any NEURALGUARD_* key set by the fleet file must map to a real
        settings field — production REFUSES unknown keys (F5).

        Wave 3: NeuralStrike's own env keys (NEURALSTRIKE_NEURALGUARD_*) are
        consumed by the neuralstrike one-shot container's pydantic settings
        (prefix NEURALSTRIKE_), NOT by the NG process — they are excluded
        from the F5 posture check by a negative lookbehind.
        """
        compose = _read(_FLEET_COMPOSE)
        keys = set(re.findall(r"(?<!NEURALSTRIKE_)NEURALGUARD_[A-Z_]+", compose))
        assert keys, "fleet compose sets no NeuralGuard env at all"
        unknown = sorted(k for k in keys if k not in known_env_keys())
        assert unknown == [], (
            f"fleet compose sets unknown env keys (F5 would refuse the boot): {unknown}"
        )

    def test_neuralguard_does_not_publish_port_8000(self) -> None:
        """Scarlet publishes :8000 — NG must serve on the fleet port."""
        compose = _read(_FLEET_COMPOSE)
        assert "${NG_FLEET_PORT:-8100}:8000" in compose, (
            "the fleet NG service must default to :8100 (Scarlet owns :8000)"
        )
        # The only other NG-port mapping is the healthcheck target (localhost
        # inside the container) — no bare "8000:8000" host binding anywhere.
        assert "8000:8000" not in compose

    def test_scarlet_stack_included_by_relative_path(self) -> None:
        compose = _read(_FLEET_COMPOSE)
        assert "include:" in compose
        assert "../../../SecurityScarletAI/docker-compose.yml" in compose, (
            "the fleet topology includes the sibling ScarletAI compose unchanged"
        )

    def test_fleet_critical_tokens_fail_fast(self) -> None:
        compose = _read(_FLEET_COMPOSE)
        for var in (
            "NEURALGUARD_AUTH_API_KEYS",
            "NEURALGUARD_CANARY_SECRET",
            "INGEST_BEARER_TOKEN",
            "MCP_BEARER_TOKEN",
        ):
            assert f"${{{var}:?" in compose, (
                f"{var} must use the :? fail-fast interpolation so a misconfigured "
                "fleet refuses to start instead of booting silently unwired"
            )

    def test_siem_and_mcp_pipes_wired_to_service_names(self) -> None:
        compose = _read(_FLEET_COMPOSE)
        # Compose-network service names of the INCLUDED Scarlet stack.
        assert "NEURALGUARD_SIEM_SCARLETAI_URL=http://api:8000/api/v1/ingest" in compose
        assert (
            "NEURALGUARD_MCP_UPSTREAM_URL=${NEURALGUARD_MCP_UPSTREAM_URL:-http://mcp:8002/mcp}"
            in compose
        )
        assert "NEURALGUARD_SIEM_ENABLED=true" in compose
        assert "NEURALGUARD_MCP_ENABLED=true" in compose
        # Wave-2: the shared bearer tokens ride both sides.
        assert "NEURALGUARD_SIEM_SCARLETAI_TOKEN=${INGEST_BEARER_TOKEN" in compose
        assert "NEURALGUARD_MCP_UPSTREAM_AUTH_TOKEN=${MCP_BEARER_TOKEN" in compose

    def test_judge_and_proxy_reach_host_ollama(self) -> None:
        compose = _read(_FLEET_COMPOSE)
        assert "NEURALGUARD_SCANNER_JUDGE_OLLAMA_URL=http://host.docker.internal:11434" in compose
        assert (
            "NEURALGUARD_PROXY_UPSTREAM_URL=${NEURALGUARD_PROXY_UPSTREAM_URL:-http://host.docker.internal:11434/v1}"
            in compose
        )

    def test_fleet_posture_is_jsonl_audit_and_memory_ratelimit(self) -> None:
        """The fleet NG runs without its own pg/redis (single worker)."""
        compose = _read(_FLEET_COMPOSE)
        assert "NEURALGUARD_AUDIT_BACKEND=jsonl" in compose
        assert "NEURALGUARD_RATELIMIT_BACKEND=memory" in compose
        assert "NEURALGUARD_RATELIMIT_REDIS_URL" not in compose

    def test_ng_service_depends_on_healthy_scarlet_api(self) -> None:
        compose = _read(_FLEET_COMPOSE)
        assert "api:" in compose.split("depends_on:")[1], (
            "NG must boot after Scarlet's API is healthy (the SIEM pipe exists at boot)"
        )

    def test_neuralstrike_service_is_profile_gated_one_shot(self) -> None:
        """The attacker joins under the bench profile, one-shot, no ports —
        `up -d` must still bring ONLY the 6 production containers."""
        compose = _read(_FLEET_COMPOSE)
        service_block = compose.split("neuralstrike:")[1].split("\nvolumes:")[0]
        assert "profiles:" in service_block
        assert "- bench" in service_block
        assert 'command: ["--help"]' in service_block, (
            "the neuralstrike service is a one-shot template — no long-lived attacker process"
        )
        # The one-shot attacker publishes NO host ports (it dials out).
        assert "ports:" not in service_block
        # The bench + purple-report share the runs volume (receipt handoff).
        assert "ns_fleet_runs:/data/runs" in service_block

    def test_neuralstrike_wiring_shares_fleet_secrets_once(self) -> None:
        """The neuralstrike service reuses the SAME fleet tokens — no secret
        is ever duplicated into a second variable."""
        compose = _read(_FLEET_COMPOSE)
        assert "NEURALSTRIKE_NEURALGUARD_URL=http://neuralguard:8000" in compose
        assert "NEURALSTRIKE_NEURALGUARD_API_KEY=${NEURALGUARD_AUTH_API_KEYS:" in compose, (
            "the screen credential is NEURALGUARD_AUTH_API_KEYS verbatim ('<key>|<tenant>' split by the bench)"
        )
        assert "NEURALSTRIKE_SCARLETAI_URL=http://api:8000/api/v1/ingest" in compose
        assert "NEURALSTRIKE_SCARLETAI_TOKEN=${INGEST_BEARER_TOKEN:" in compose
        assert "NEURALSTRIKE_SCARLETAI_BASE_URL=http://api:8000" in compose
        assert "NEURALSTRIKE_SCARLETAI_API_TOKEN=${API_BEARER_TOKEN:" in compose
        # The bench drives no LLM — the container must skip the reachability
        # check instead of stalling on an Ollama it does not need.
        assert "NEURALSTRIKE_SKIP_REACHABILITY_CHECK=true" in compose

    def test_neuralstrike_builds_from_sibling_checkout(self) -> None:
        compose = _read(_FLEET_COMPOSE)
        assert "context: ../../../NeuralStrike" in compose, (
            "the neuralstrike service builds from the sibling NS checkout (same layout as the Scarlet include)"
        )

    def test_neuralstrike_service_depends_on_healthy_ng_and_api(self) -> None:
        compose = _read(_FLEET_COMPOSE)
        ns_block = compose.split("neuralstrike:")[1].split("\nvolumes:")[0]
        assert "neuralguard:" in ns_block.split("depends_on:")[1]
        assert "api:" in ns_block.split("depends_on:")[1]
        assert "service_healthy" in ns_block

    def test_env_template_lists_every_required_secret(self) -> None:
        example = _read(_FLEET_ENV_EXAMPLE)
        compose = _read(_FLEET_COMPOSE)
        for var in (
            "DB_PASSWORD",
            "API_SECRET_KEY",
            "API_BEARER_TOKEN",
            "DB_READONLY_PASSWORD",
            "MCP_BEARER_TOKEN",
            "INGEST_BEARER_TOKEN",
            "NEURALGUARD_AUTH_API_KEYS",
            "NEURALGUARD_CANARY_SECRET",
        ):
            assert f"{var}=" in example, f"{var} missing from fleet.env.example"
        # SECRET vars must be CHANGE_ME placeholders — never real-looking values;
        # non-secret tunables (ports, model tags) may carry their defaults.
        for line in example.splitlines():
            if not re.match(r"^[A-Z_]+=", line):
                continue
            name, value = line.split("=", 1)
            if name in {
                "DB_PASSWORD",
                "API_SECRET_KEY",
                "API_BEARER_TOKEN",
                "DB_READONLY_PASSWORD",
                "MCP_BEARER_TOKEN",
                "INGEST_BEARER_TOKEN",
                "NEURALGUARD_AUTH_API_KEYS",
                "NEURALGUARD_CANARY_SECRET",
            }:
                assert "CHANGE_ME" in value, (
                    f"{name} must stay a CHANGE_ME placeholder in the template"
                )
