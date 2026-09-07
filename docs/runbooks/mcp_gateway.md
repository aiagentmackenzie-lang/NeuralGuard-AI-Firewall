# Runbook — MCP Gateway (NG-7 / NG-8)

*Shipped 2026-09-07. The MCP gateway turns NeuralGuard into an intent-gated,
rug-pull-refusing passthrough in front of ONE MCP server (streamable-HTTP
transport): `POST /v1/mcp`.*

## What it enforces

1. **NG-8 Intent Gate (pre-body-parse).** `Mcp-Method` (and `Mcp-Name` on
   `tools/call`) headers are evaluated against the tenant's per-tool /
   per-method policy BEFORE the body is parsed. DENY → 403, upstream never
   called. ESCALATE → 403 + audit (HITL: an approval workflow is a follow-up;
   this build never silently passes a gated intent).
2. **Smuggling defense.** Body intent must agree with header intent — a
   mismatch (`MCP-GATE-SMUGGLE-001/002`) blocks.
3. **NG-7 rug-pull refusal.** The tool catalog from every `tools/list` is
   hashed (canonical: sorted tools, name+description+inputSchema) against a
   signed baseline:
   - strict (default): a drifted catalog is WITHHELD (403 +
     `X-NeuralGuard-Mcp-Drift: 1` + drift report) and the gateway refuses
     EVERY `tools/call` until an explicit re-baseline.
   - advisory: drift is alerted (`X-NeuralGuard-Mcp-Drift: advisory` + audit
     event) but the catalog passes — canary deployments only.
   - A `tools/call` for a tool the baseline NEVER contained is refused in
     every mode (`MCP-RUGPULL-002`) — that is the payload executing, not
     "drift to observe".

## Configuration (all `NEURALGUARD_MCP_*`)

| Key | Default | Meaning |
|---|---|---|
| `ENABLED` | `false` | Mount the `/v1/mcp` routes. |
| `UPSTREAM_URL` | — | The MCP server's JSON-RPC endpoint. Required when enabled. |
| `MODE` | `strict` | `strict` \| `advisory` (see above). |
| `SERVER_ID` | `default` | Audit/baseline key. One gateway = one upstream in this build. |
| `SIGNING_SEED` | — | Ed25519 seed (hex) to SIGN baselines (P2-10 crypto). Empty = unsigned. **Secret — rotate via `secret_rotation.md`.** |
| `VERIFY_PUBKEY` | — | Ed25519 pubkey (hex) to verify registry-signed catalog changes (the `X-MCP-Catalog-Signature` recovery path). |
| `REQUIRE_SIGNATURE_ON_CHANGE` | `false` | When true, drift resolves ONLY via a verified signature. |
| `HEADERS_REQUIRED` | `true` | The NG-8 seam. Turning this off disables the pre-parse gate (logged). |

### Tenant policies (NG-8)

Per-tenant files (`tenants/<tenant>.yaml`) accept an `mcp:` block:

```yaml
tenant_id: acme
mcp:
  tool_rules:
    delete_file: deny
    deploy_prod: escalate
  method_rules:
    resources/list: allow
  default_tool_action: allow   # deny for a default-deny tenant
```

Most-restrictive-wins when both a method rule and a tool rule apply
(DENY > ESCALATE > ALLOW).

## Drift recovery (the operator's part)

When strict mode poisons the gateway (drift detected):

1. **Verify the drift is legitimate.** Every drift event in the audit trail
   carries `old_catalog_hash` + `new_catalog_hash` + the added/removed/changed
   tool diff. Diff the catalogs for real.
2. **Legitimate change** (planned server upgrade): either
   - have the registry sign the new catalog and pin `VERIFY_PUBKEY` — the
     next `tools/list` with `X-MCP-Catalog-Signature` auto-recovers, or
   - restart the worker / bump config and let `rebaseline` run — logged as
     `REBASELINED` (explicit) or `BASELINE_RECREATED_RESTART` (restart).
3. **Unexpected change**: treat as an incident. The baseline state machine
   refused every tool call in the meantime — containment was automatic.

## Honest limits (documented, not hidden)

- **In-memory baselines per worker.** Restart → `BASELINE_RECREATED_RESTART`
  (loud audit marker). Drift evidence survives restarts: every historical
  hash pair is in the hash-chained audit trail.
- **One upstream per gateway.** Multi-server routing is future work; run one
  gateway per MCP server (compose service) meanwhile.
- **JSON responses only.** SSE/streaming MCP sessions are refused (422
  posture parity with the chat proxy) — unscanned streams must not pass.
- **Tool RESULT content** is forwarded as-is in this build; scanning tool
  output through the existing output-scan semantics is the immediate
  follow-up (the indirect-injection-in-tool-results class).
- **ESCALATE = refuse + audit** in this build. A human-approval callback
  flow is future work.

## Verification drill

```bash
# 1. Enable with a fake upstream and run the suite:
NEURALGUARD_MCP_ENABLED=true NEURALGUARD_MCP_UPSTREAM_URL=http://localhost:9999 \
  uv run pytest tests/unit/test_mcp_gateway.py -q --no-cov

# 2. Metrics: decisions + baseline outcomes are exported.
#    neuralguard_mcp_gate_total{action=...}
#    neuralguard_mcp_baseline_total{outcome=...}

# 3. Audit: every gate decision lands in the SAME hash-chained trail
#    (audit-verify covers MCP events — no separate tooling).
```