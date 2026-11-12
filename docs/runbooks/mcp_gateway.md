# Runbook — MCP Gateway (NG-7 / NG-8 / NG-10 / NG-11)

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
| `UPSTREAM_AUTH_TOKEN` | — | Server-side bearer token forwarded as the upstream's `Authorization: Bearer <token>` (e.g. SecurityScarletAI's `MCP_BEARER_TOKEN`). Empty = no auth header (legacy local-MCP posture). **Secret — never logged; gateway callers cannot override it; rotate via `secret_rotation.md`.** |

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

### Argument and resource constraints (NG-10)

The Intent Gate says *whether* a tool may run; `tool_constraints` says *how* it
may be asked. Constraints are evaluated on the parsed `params.arguments` of a
`tools/call`, before the upstream is contacted, and every configured rule is
evaluated so the strictest violation wins (DENY > ESCALATE > ALLOW) regardless
of declaration order.

```yaml
tenant_id: acme
mcp:
  tool_rules:
    delete_file: deny
  tool_constraints:
    read_file:
      - argument: path
        kind: required
      - argument: path
        kind: path_prefix
        values: ["/srv/tenant-data"]
        reason: "tenant data lives under /srv/tenant-data"
    http_get:
      - argument: url
        kind: url_host
        values: ["api.internal.example"]      # host + subdomains
    run_query:
      - argument: ""
        kind: key_allowlist                   # top-level keys only
        values: ["query", "limit"]
      - argument: "query"
        kind: regex_deny
        values: ["(?i)\\b(drop|truncate|grant)\\b"]
      - argument: "limit"
        kind: max_length
        values: ["4"]
```

Kinds: `required`, `key_allowlist`, `enum`, `max_length`, `regex_deny`,
`path_prefix`, `path_glob`, `url_host`. `argument` is a dotted path
(`options.format`); `""` addresses the arguments object (key_allowlist).
Set `action: escalate` to refuse-but-audit instead of a hard DENY.

Semantics worth knowing before writing a policy:

- **Absent is vacuous** — a rule for an argument that is not present does not
  fire. Add `required` when the argument must exist. Without this, every
  optional argument would need a paired rule and one omission would widen the
  policy.
- **Non-strings fail closed** — a list or boolean where a path/URL is expected
  is `MCP-ARG-009`, not a skipped check.
- **Path checks normalize first** — `..` segments, backslashes and NUL bytes
  are refused outright; prefix matching is boundary-aware (`/srv/data-evil`
  does not match `/srv/data`).
- **URL host checks resolve the real host** — scheme must be http(s);
  `https://allowed.example@evil.test/` is evaluated as `evil.test`.
- **Values never reach the audit trail** — findings carry the argument path,
  the rule id, and the failure kind. `key_allowlist` names the extra keys, not
  their values.

This is a boundary check on declared arguments, **not a sandbox**: it stops a
confused or coerced agent from asking for the wrong thing. It does not confine
an upstream tool that ignores its own arguments, and path checks are
string-level — no symlink resolution.

### Workload identity and per-agent policy (NG-11)

Every caller already authenticates as a tenant. Identity now also names the
**workload agent** inside that tenant, and the agent can only ever *narrow* what
the tenant allows.

The identity comes from the credential, never from a header:

```bash
# config: a key can be bound to one agent
NEURALGUARD_AUTH_API_KEYS="key-abc...|acme|support-bot,key-def...|acme"
#                                 ^tenant ^agent      ^tenant only (no agent)
```

A JWT minted in exchange for that key carries the binding as an `agent` claim,
so it survives token exchange. **A caller cannot request an agent** — there is
no field for it on `POST /v1/auth/token`. A header claiming to be an agent is
ignored, which is the whole point: otherwise every per-agent rule would be
decorative.

```yaml
tenant_id: acme
mcp:
  tool_rules:
    read_file: allow          # tenant-wide
  agent_rules:
    support-bot:
      tool_rules:
        delete_file: deny     # this bot may not delete
      tool_constraints:       # stacked ON TOP of the tenant's
        read_file:
          - argument: path
            kind: path_prefix
            values: ["/srv/support"]
    report-writer:
      tool_rules:
        read_file: deny
  # Callers whose agent is not listed here (including tenant-level keys):
  default_agent_action: allow   # set deny to enumerate your fleet strictly
```

Resolution rules, all fail-closed:

- **Narrowing only.** The strictest of the tenant rule and the agent rule wins
  (DENY > ESCALATE > ALLOW). An agent rule can deny what the tenant allows; it
  can never allow what the tenant denies, and cannot downgrade ESCALATE.
- **Constraints stack.** Agent constraints are evaluated *in addition to* the
  tenant's — a violation of either denies. There is no way to remove a tenant
  constraint from an agent policy.
- **Agent rules are tool-scoped.** They apply to `tools/call`; method rules
  stay a tenant-level concern.
- **Unlisted is not "any agent".** Once `agent_rules` exists, a caller with an
  unknown or absent agent is governed by `default_agent_action` (default
  `allow`, so adding rules for one bot does not lock out everyone else).
- **Every audit event carries `agent_id`** — explicitly `null` for tenant-level
  credentials, so "which agent did this" is answerable from the trail alone.

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
- **NG-10 is not confinement.** Argument constraints are string-level boundary
  checks on what the caller *declares*. They do not sandbox the upstream tool,
  do not resolve symlinks, and do not cover arguments a tool derives on its
  own. Tool-result scanning (above) remains the other half of the story.
- **Identity is per credential, not per process.** Two workloads sharing one
  key are one agent; per-agent policy is only as precise as the key hygiene
  behind it. Issue one key per agent (the rotation API mints tenant-level keys
  only, so agent-bound keys currently come from configuration).
- **Scoped credentials are still missing.** One `UPSTREAM_AUTH_TOKEN` serves
  every tool: identity narrows *what* an agent may call, not *which upstream
  credential* the call carries. Per-tool credential selection is the next
  gate, and it stays a server-side mapping — never a caller-supplied header.

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