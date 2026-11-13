# Runbook — MCP Gateway (NG-7 / NG-8 / NG-10 / NG-11 / NG-12 / NG-14 / NG-15)

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

### Scoped upstream credentials, per tool and per agent (NG-12 / NG-14)

One token for every tool meant the upstream could not tell a read from a
delete (NG-12). One token per *tool* still meant two agents permitted that tool
shared an upstream principal, so the upstream could not say **which agent**
made the call. A mapping key is now `tool` (tool-wide) or `agent|tool`:

```bash
# JSON in the env var. REFERENCES only — a secret pasted here fails at boot.
NEURALGUARD_MCP_TOOL_CREDENTIALS='{
  "read_file":            "env:READONLY_TOKEN",
  "billing-bot|refund":   "file:/run/secrets/refund-billing",
  "support-bot|refund":   "env:REFUND_SUPPORT"
}'
NEURALGUARD_MCP_REQUIRE_TOOL_CREDENTIAL=true   # optional: refuse unmapped tools
```

The agent half is the **authenticated** NG-11 identity (a `key|tenant|agent`
binding or a JWT `agent` claim) — never a header, and normalized/validated at
boot (lowercased, `[a-z0-9._-]`, 1–64 chars), exactly like the policy key.

Resolution order for `tools/call`, most specific first:

1. **`(agent, tool)`** → that agent's own reference, resolved per call.
2. **`tool`** → the tool-wide reference (any agent, or a tenant-level caller).
3. **No mapping and no default fallback available** → refuse if the tool is
   agent-scoped, or if `REQUIRE_TOOL_CREDENTIAL=true`; otherwise the default.
4. **Catalog methods** (`tools/list`, …) → the default token, as before.

Fail-closed rules:

- **A mapped reference that cannot be resolved refuses the call** (`503
  mcp_credential_unavailable`, `MCP-CRED-001`) — it never falls back to the
  default. Falling back would hand a deliberately-scoped call the broad token.
- **An agent-scoped tool never degrades to the default.** Once any
  `agent|tool` entry exists for a tool, a caller with no matching entry is
  refused — including a tenant-level (agent-less) credential. Otherwise
  configuring `billing-bot|refund` would hand every other agent the broad
  default for `refund`, which is exactly the sharing this removes. Add a
  `tool` key when the fallback is genuinely intended.
- **The denial is attributable.** The refused audit event records
  `agent_id`, `credential_source: "agent-scoped-unmapped"` and a reason naming
  the tool and agent — so "why was billing refused and support allowed" is
  answerable from the trail.
- **Secrets are read per call**, not cached at boot, so an orchestrator can
  mount/project them after the process starts. Unresolvable references are
  listed **by name** in a boot warning (`mcp_credential_unresolved`), with
  `agent_scoped_tools` so the operator can see which tools lost their fallback.
- **Only the reference reaches the audit trail** — never the value. The
  forwarded event records `credential_source: "env:READONLY_TOKEN"` alongside
  `agent_id`.
- **A caller can never supply its own upstream token.** The transport strips
  any `Authorization` header the caller sent (any casing) before setting the
  server-side one.

Stated limits, not hidden:

- `file:` references read whatever the process can read. This module checks the
  secret is non-empty and under 4 KiB (so a mis-pointed PEM bundle fails
  loudly); it does **not** police filesystem permissions — use a 0600 mount.
- Agent-scoped credentials select *which upstream principal* carries the call;
  they do not constrain *what* the agent may ask (NG-10/NG-11 do that). The
  operator must still make the upstream principals differ in authority — two
  agent entries pointing at the same token are audited, but attribution is
  only as strong as the upstream's own logging.

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

## NG-15 — Human-in-the-loop approval for ESCALATE

An `ESCALATE` verdict means "a human must approve this". Before NG-15 the
gateway refused the call and left the approval to a human reading the audit
trail. Now the loop is closed, on the same fail-closed terms.

```bash
NEURALGUARD_MCP_HITL_ENABLED=true
NEURALGUARD_MCP_HITL_TIMEOUT_SECONDS=10        # 1-60s the request may be held
NEURALGUARD_MCP_HITL_TTL_SECONDS=300           # pending lifetime (>= timeout)
NEURALGUARD_MCP_HITL_MAX_PENDING=256           # per worker; at the cap, refuse
NEURALGUARD_MCP_HITL_APPROVER_TENANTS=admin    # empty = the admin tenant only
#   A named approver tenant decides only ITS OWN escalations; the admin tenant
#   (the gateway operator) decides any. That keeps the approver list from
#   becoming a cross-tenant read of another tenant's tool arguments.
```

Flow for a `tools/call` the tenant policy escalates:

1. The gateway raises an escalation bound to the **exact** call — tenant,
   agent, tool, method and a SHA-256 digest of the canonical JSON-RPC body.
   A smuggling mismatch is blocked *before* the human is asked.
2. It waits, bounded by `HITL_TIMEOUT_SECONDS`, for a decision.
3. An approver decides via the API (below).
4. **Approved** → the request continues through every remaining gate: argument
   constraints, the signed-catalog baseliner, the provenance gate and the
   credential resolver. **Denied, timed out, expired or capacity-refused** →
   the call is refused, exactly as before.

### Approver API

```bash
# What is waiting (oldest first; ids, digest, bounded argument preview)
curl -H "Authorization: Bearer $APPROVER_KEY" https://gateway/v1/mcp/escalations

# Decide — single-shot, audited, attributed
curl -X POST -H "Authorization: Bearer $APPROVER_KEY"   -d '{"approve": true, "reason": "change window CHG-1041"}'   https://gateway/v1/mcp/escalations/<escalation_id>
```

Responses: `200` decision recorded · `400` `invalid_decision` · `403`
`mcp_escalation_not_approver` / `mcp_escalation_agent_forbidden` /
`mcp_escalation_other_tenant` · `404` `mcp_escalation_unknown` (unknown or
already consumed) · `409` `mcp_escalation_not_pending` · `503`
`mcp_hitl_disabled`.

The waiting caller sees one of `mcp_escalation_denied` /
`mcp_escalation_timeout` / `mcp_escalation_capacity` — all three are a `403`
(or `503` for capacity), never a pass.

Audit events: `MCP-HITL-RAISED`, `MCP-HITL-APPROVED`, `MCP-HITL-DENIED`,
`MCP-HITL-TIMEOUT`, `MCP-HITL-CAPACITY`, and `MCP-HITL-DECISION` on the
approver's action. Every one carries the escalation id, the request digest,
the source gate (`intent_gate` / `argument_gate`), `agent_id`, and — for
decisions — `decided_by`. Argument **values** never enter the audit trail; the
digest is the binding. The approver view carries a bounded argument preview
because reviewing the call is the point of the review.

### Fail-closed rules

- **A decision requires a tenant-level credential.** An agent-bound credential
  (NG-11) is refused with `mcp_escalation_agent_forbidden`: an agent raising an
  escalation for itself and approving it is the abuse case this feature would
  otherwise create.
- **Only named approvers decide, and only inside their own tenant.** The
  deciding tenant must be in `HITL_APPROVER_TENANTS` (default: the admin
  tenant), and a non-admin approver tenant may only see and decide its **own**
  escalations (`403 mcp_escalation_other_tenant`). The admin tenant is global
  because it is the gateway operator. No approver configured means no approval
  path — `hitl_enabled` is off by default and the old refusal is unchanged.
- **Approval is single-use and digest-bound.** An approval authorizes exactly
  the request that raised it; the id is consumed on use and a replay finds
  nothing. It cannot be applied to different arguments.
- **Approval is not authority.** A human "yes" cannot make a rug-pull execute:
  the baseliner still refuses an unbaselined tool, the argument gate still
  denies, the provenance gate still blocks, and the credential resolver still
  fails closed.
- **Timeouts and capacity are refusals.** There is no path from an unanswered
  escalation to an approved call.

### Operability limits

- **Per worker, in memory.** A decision must reach the worker holding the
  pending request. With more than one replica, route
  `/v1/mcp/escalations` with sticky sessions or run one replica; a shared
  broker is not in this build (same shape as the per-worker baselines).
- **A held request is a resource.** The wait is capped at 60 s and pending
  records at `HITL_MAX_PENDING`; at the cap a new `ESCALATE` is refused
  (`503 mcp_escalation_capacity`) rather than queued.
- **Notification is the SIEM's job.** The gateway does not page anyone: the
  raised escalation lands in the audit chain, which your existing SIEM routing
  (Splunk HEC / webhook / SecurityScarletAI) already forwards. Approvers poll
  or consume that alert; the decision API is the callback surface.
- **Capacity is global, not per tenant.** One noisy tenant can fill
  `HITL_MAX_PENDING` and cause other tenants' escalations to be refused
  (`503 mcp_escalation_capacity`). That is a deliberate fail-closed refusal —
  nothing is auto-approved — but it is an availability coupling, so size the
  cap for the noisiest tenant you expect.

### What "human" means here (stated, not implied)

The gateway authenticates a **credential**, not a person: `decided_by` records
the approver *tenant*, and a script holding that tenant's key is
indistinguishable from a human at a keyboard. What the control does guarantee
is the property that matters for containment — the decision cannot come from
the workload agent's own credential, cannot be replayed, cannot be applied to a
different payload, and cannot be granted by timeout. If you need
person-level approval, put the approver key behind your IdP/SSO proxy and let
that layer be the identity boundary.

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
- **HITL is per worker and in memory.** An approval only reaches the worker
  holding the pending call (sticky routing or a single replica); the request is
  held for at most `HITL_TIMEOUT_SECONDS`, and notification is delegated to the
  SIEM routing of the raised audit event.
- **NG-10 is not confinement.** Argument constraints are string-level boundary
  checks on what the caller *declares*. They do not sandbox the upstream tool,
  do not resolve symlinks, and do not cover arguments a tool derives on its
  own. Tool-result scanning (above) remains the other half of the story.
- **Identity is per credential, not per process.** Two workloads sharing one
  key are one agent; per-agent policy is only as precise as the key hygiene
  behind it. Issue one key per agent (the rotation API mints tenant-level keys
  only, so agent-bound keys currently come from configuration).
- **Scoped credentials select a principal, not authority.** Per-(agent, tool)
  credentials (NG-14) let the upstream attribute a call to the agent that made
  it, but two agent entries pointing at the same token are attributable without
  being distinct authority — the upstream's own authorization still decides.

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

### Drill — the HITL loop (NG-15)

Runs against a gateway whose tenant policy escalates one tool (the automated
suite covers the same loop through the real ASGI app in
`tests/unit/test_mcp_hitl.py`; this is the operator's version).

```bash
# 1. A tenant whose policy escalates the tool. Written to /tmp on purpose —
#    a drill file inside the repo would show up in `git status`.
mkdir -p /tmp/ng-tenants
cat > /tmp/ng-tenants/acme.yaml <<'YAML'
tenant_id: acme
mcp:
  tool_rules:
    deploy_prod: escalate
YAML

# 2. Enable tenants + HITL, with a caller bound to an agent and an admin
#    approver key, and a 10 s wait.
export CALLER_KEY=callerkey APPROVER_KEY=approverkey
export NEURALGUARD_TENANT_ENABLED=true NEURALGUARD_TENANT_CONFIG_PATH=/tmp/ng-tenants
export NEURALGUARD_AUTH_ENABLED=true
export NEURALGUARD_AUTH_API_KEYS="callerkey|acme|deploy-bot,approverkey|admin"
NEURALGUARD_MCP_ENABLED=true NEURALGUARD_MCP_UPSTREAM_URL=http://localhost:9999 \
NEURALGUARD_MCP_HITL_ENABLED=true NEURALGUARD_MCP_HITL_TIMEOUT_SECONDS=10 \
NEURALGUARD_MCP_HITL_APPROVER_TENANTS=admin \
  uv run neuralguard serve --port 8000

# 3. Call the escalated tool. It BLOCKS for up to 10 s; in another shell:
curl -s -H "Authorization: Bearer $CALLER_KEY" -H "Mcp-Method: tools/call" \
  -H "Mcp-Name: deploy_prod" -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
  "params":{"name":"deploy_prod","arguments":{"env":"prod"}}}' \
  http://localhost:8000/v1/mcp &

# 4. Approve it (or deny, or do nothing and watch it time out).
curl -s -H "Authorization: Bearer $APPROVER_KEY" \
  http://localhost:8000/v1/mcp/escalations
curl -s -H "Authorization: Bearer $APPROVER_KEY" \
  -d '{"approve":true,"reason":"CHG-1041"}' \
  http://localhost:8000/v1/mcp/escalations/<id>

# 5. Expect: 200 forwarded + MCP-HITL-RAISED then MCP-HITL-APPROVED in the trail
#    on approval; 403 mcp_escalation_denied / _timeout with nothing forwarded
#    otherwise; 403 mcp_escalation_agent_forbidden if the approver key is
#    agent-bound; 403 mcp_escalation_other_tenant for another tenant's call.
uv run neuralguard audit-verify ./audit_logs
```