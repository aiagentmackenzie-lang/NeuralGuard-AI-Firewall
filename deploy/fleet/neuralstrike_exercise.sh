#!/usr/bin/env bash
# NeuralStrike purple-team exercise (fleet Wave 3) — the third pipe:
#
#   NeuralStrike attacks → NeuralGuard defends → SecurityScarletAI detects →
#   the purple-report measures what the SIEM actually caught.
#
# Runs the ONE-SHOT neuralstrike service twice (profile `bench`):
#   1. neuralguard-bench  — telemetry ON (exercise_start → probes →
#      exercise_end into Scarlet via the shared INGEST_BEARER_TOKEN);
#      receipt into the ns_fleet_runs volume.
#   2. purple-report      — reads the receipt + the SIEM (admin-class
#      API_BEARER_TOKEN, read-only /alerts + /logs) and prints the
#      detection-coverage report (X attacks, Y caught, Z alerts, the
#      per-payload table, and the UNDETECTED-SUCCEEDED gap list).
#
# Requires: the fleet compose UP (see the runbook), the sibling NeuralStrike
# checkout at ../../../NeuralStrike, and deploy/fleet/.env filled.
#
# Usage:  bash deploy/fleet/neuralstrike_exercise.sh
# Env:    SETTLE_SECONDS (default 45 — correlation/rules settle window),
#         EXERCISE_LABEL (default timestamp), NS_SERVICE (the compose service)

set -uo pipefail
cd "$(dirname "$0")"

SETTLE_SECONDS="${SETTLE_SECONDS:-45}"
SCARLET_HEALTH_TIMEOUT=10

# Fleet env (generated secrets; NEVER committed). Parsed per-key (not sourced)
# so values with shell-special characters cannot break the harness.
if [[ ! -f .env ]]; then
  echo "FATAL: deploy/fleet/.env missing — copy fleet.env.example and fill it" >&2
  exit 1
fi
_env() { grep -E "^$1=" .env | head -1 | cut -d= -f2-; }
NG_CREDENTIAL="$(_env NEURALGUARD_AUTH_API_KEYS)"
INGEST_TOKEN="$(_env INGEST_BEARER_TOKEN)"
API_TOKEN="$(_env API_BEARER_TOKEN)"
[[ -n "$NG_CREDENTIAL" && -n "$INGEST_TOKEN" && -n "$API_TOKEN" ]] || {
  echo "FATAL: NEURALGUARD_AUTH_API_KEYS / INGEST_BEARER_TOKEN / API_BEARER_TOKEN must be set in .env" >&2
  exit 1
}
NG_TENANT="${NEURALSTRIKE_NEURALGUARD_TENANT:-$(echo "$NG_CREDENTIAL" | cut -d'|' -f2)}"
NG_TENANT="${NG_TENANT:-default}"
NG_URL="http://neuralguard:8000"
INGEST_URL="http://api:8000/api/v1/ingest"
COMPOSE=(docker compose -f docker-compose.fleet.yml)

stamp="$(date +%Y%m%d%H%M%S)"
receipt="/data/runs/exercise-${stamp}.json"
report="/data/runs/purple-${stamp}.json"

echo "== Phase 0: health gates (fail-closed — a dark stack proves nothing)"
scarlet_health=$(curl -sf --max-time "$SCARLET_HEALTH_TIMEOUT" http://127.0.0.1:8000/api/v1/health || echo "")
[[ "$scarlet_health" == *'"healthy"'* || "$scarlet_health" == *'"degraded"'* ]] || {
  echo "FATAL: ScarletAI API unhealthy on :8000 — bring the fleet up first (see the runbook)" >&2
  exit 1
}
echo "  ✅ ScarletAI API up"

echo "== Phase 1: the exercise (telemetry ON; receipt → ${receipt})"
# The one-shot container reads the fleet env wiring from the compose service;
# the flags here mirror it so the receipt names exactly what ran. The screen
# credential is NEURALGUARD_AUTH_API_KEYS verbatim — the bench splits the
# documented '<key>|<tenant>' form (resolve_neuralguard_credential).
"${COMPOSE[@]}" run --rm neuralstrike \
  neuralguard-bench \
  --neuralguard-url http://neuralguard:8000 \
  --neuralguard-api-key "$NG_CREDENTIAL" \
  --neuralguard-tenant "$NG_TENANT" \
  --scarletai-url "$INGEST_URL" \
  --scarletai-token "$INGEST_TOKEN" \
  --telemetry-actor "${NEURALSTRIKE_TELEMETRY_ACTOR:-neuralstrike-operator}" \
  --json-out "$receipt" || { echo "FATAL: the bench failed — no exercise to report on" >&2; exit 1; }
echo "  ✅ exercise complete; receipt at ${receipt} (ns_fleet_runs volume)"

echo "== Phase 2: settle window (${SETTLE_SECONDS}s — rules + correlation settle)"
sleep "$SETTLE_SECONDS"

echo "== Phase 3: the purple report"
run_id="$(basename "$receipt" .json | sed 's/^exercise-//')"
"${COMPOSE[@]}" run --rm neuralstrike \
  purple-report "$receipt" \
  --scarlet-base-url http://api:8000 \
  --scarlet-api-token "$API_TOKEN" \
  --ng-tenant "$NG_TENANT" \
  --json-out "$report" || { echo "FATAL: purple-report failed (LOUD — see above)" >&2; exit 1; }
echo "  ✅ purple report at ${report} (ns_fleet_runs volume)"
echo "== Done. Copy artifacts out of the volume if needed:"
echo "   docker compose -f docker-compose.fleet.yml run --rm --entrypoint cat neuralstrike /data/runs/$(basename "$receipt")"