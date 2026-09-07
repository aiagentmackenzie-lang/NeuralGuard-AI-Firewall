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

## NG-9 — Provenance-lite egress binding (opt-in)

The final frontier control: tool-result content passing through the gateway
TAINTS the session window (keyed on the spec's `Mcp-Session-Id` header), and
calls to tenant-classified **egress tools** get their arguments checked for
tainted content BEFORE forwarding.

- Classification is per-tenant (`mcp.egress_tools` in the tenant file — the
  tenant knows its own tools); the posture is operator-global:
  `NEURALGUARD_MCP_PROVENANCE_MODE` = `off` (default) | `warn` | `block`.
- Matching is over **normalized 4-word shingles** + long-token (≥32 char)
  prefix fingerprints — partial quotes of ≥4 contiguous words and verbatim
  base64/hex blobs are caught; case/whitespace/zero-width tricks do not
  evade (the comparison form strips them).
- `warn` = alert + allow (audit event `provenance_warned` +
  `X-NeuralGuard-Mcp-Provenance: tainted-warn`); `block` = 403
  `MCP-PROV-001` (tainted arguments never leave the trust boundary).
- Fail-closed edge: `NEURALGUARD_MCP_PROVENANCE_REQUIRE_SESSION=true` +
  mode on → an egress call with NO session header is REFUSED
  (`MCP-PROV-002`) — un-attributable calls do not bypass taint attribution.
  Without it, session-less clients share the tenant-level taint window.
- Sessions are LRU-bounded (`PROVENANCE_MAX_SESSIONS`) and TTL'd
  (`PROVENANCE_TTL_SECONDS`); in-memory per worker (same posture as the
  baselines).

**Detection boundary (honest):** word-order scrambling that breaks every
contiguous 4-word sequence, and paraphrase generally, are NOT caught — that
is CaMeL-class data-flow enforcement, deliberately out of scope for a
middleware. What NG-9 stops is the dominant real shape: tool-output content
(p planted instructions, secrets, dumped configs) flowing verbatim into an
outbound tool call.

**Ordering guarantee (hardened after the NG-9 integration tests caught a
route-ordering regression):** tools/call baseline + provenance checks run
BEFORE the upstream forward — a refused call is never executed. Only
`tools/list` forwards before its check (you cannot baseline a catalog you
have not fetched).

## Honest limits (documented, not hidden)

- **In-memory baselines per worker.** Restart → `BASELINE_RECREATED_RESTART`
  (loud audit marker). Drift evidence survives restarts: every historical
  hash pair is in the hash-chained audit trail.
- **One upstream per gateway.** Multi-server routing is future work; run one
  gateway per MCP server (compose service) meanwhile.
- **JSON responses only.** SSE/streaming MCP sessions are refused (422
  posture parity with the chat proxy) — unscanned streams must not pass.
- **Tool RESULT content** is taint-fingerprinted (NG-9) but not yet
  scanned through the output-scan pattern semantics — that remains the
  immediate follow-up for the indirect-injection-in-tool-results class.
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