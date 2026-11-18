# Secret Rotation Runbook (P0-3)

Secrets in active use (rotation procedures below):

- **API keys** (`NEURALGUARD_AUTH_API_KEYS`) — authenticate clients and bind
  them to tenants. Runtime rotation is ALSO available via the P2-4 rotation
  API (see §Runtime rotation API).
- **Postgres password** (`POSTGRES_PASSWORD` / the password embedded in
  `NEURALGUARD_AUDIT_POSTGRES_URL`, when `audit.backend=postgres`).
- **JWT signing secret** (`NEURALGUARD_AUTH_JWT_SECRET`, when JWT auth is
  enabled) — rotating it invalidates all outstanding tokens immediately.
- **Canary secret** (`NEURALGUARD_CANARY_SECRET`) — rotating it invalidates
  all outstanding canary tokens (do it on any suspected leak; that is the
  documented response path).
- **Audit signing key** (`NEURALGUARD_AUDIT_SIGNING_KEY`, Ed25519 seed hex,
  P2-10) — rotates by key epoch; verify per epoch with `audit-verify
  --pubkey`.
- **Redis password** (`REDIS_PASSWORD` in the appliance compose — requirepass
  protects the shared rate-limit + Agent Guardian session state).

Source of truth for secrets: a secret manager (SOPS, Vault, AWS Secrets
Manager, GCP Secret Manager). `.env` is for local dev only — never commit
real secrets (`.gitignore` already excludes `.env`).

## Rotating API keys (zero downtime)

The key format is `key|tenant` (bare key → tenant `default`). A key is
valid as long as it appears in `NEURALGUARD_AUTH_API_KEYS`. Adding a new
key takes effect on the next process start; removing one takes effect on
the next start. So rotation = add new + restart + verify + remove old +
restart.

### 1. Issue the new key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# e.g. nw_K8sQ...-newkey
```

### 2. Dual-key window (add new, keep old)

Set both keys in the env, comma-separated, bound to the same tenant:

```bash
NEURALGUARD_AUTH_API_KEYS=old-key-...|acme,new-key-...|acme
```

Rolling-restart the workers (one at a time). Both keys now authenticate.
**Do not** remove the old key yet — in-flight clients still using it must
not get 401s.

### 3. Migrate clients

Distribute the new key to each client out-of-band (not over the same
channel the key protects). Clients move at their own pace; the dual-key
window keeps both working.

### 4. Verify no traffic uses the old key

Watch auth rejections + a sampled audit log for the old key's tenant. The
audit log records `tenant_id`, not the key itself. To confirm the old key
is idle, temporarily rely on metrics:

```bash
curl -s -H "X-API-Key: new-key-..." http://host/v1/metrics | grep auth_rejections
```

When you are confident no client uses the old key (give it a full business
cycle — 24h minimum), proceed.

### 5. Remove the old key

```bash
NEURALGUARD_AUTH_API_KEYS=new-key-...|acme
```

Rolling-restart. The old key now returns 401. Keep the old key in the
secret manager (marked retired) for one more cycle in case of rollback,
then purge it.

### Forced rotation (incident)

If a key is known compromised, skip the dual-key window: set only the new
key and restart immediately. In-flight clients using the old key get 401s
until they rotate — that is the intended fail-closed behavior for a
compromised credential.

**But a rolling restart is not immediate revocation.** Measured with two
replicas: the revoked key remained valid for **21 s** (~23 % of sampled requests
still succeeded) because the Service kept routing to pods that had not rolled
yet. With `strategy: Recreate` the revoked key was valid for **0 s** at the cost
of **~7 s** of total unavailability. Choose deliberately — see "Drilled" below.

## Drilled: what the rotation windows actually are (two replicas, 2026-11-17)

The env/redeploy path above was recommended but never measured under two
replicas. It is now measured — kind v0.33.0, 2 `neuralguard` pods behind a
Service, a sampler in a separate pod probing the Service ~2×/second (so every
number below is **client-visible**, not pod-internal), and the per-pod posture
(`GET /v1/info` → `auth.key_count` / `auth.key_set_id`) read on each pod.

Environment: `environment=production`, `RollingUpdate` (maxSurge 1 /
maxUnavailable 0) unless stated, image already on the node, rate limit raised
for the drill only (the 120/min default refuses a probe loop — left at 120 the
drill measures the rate limiter instead).

### Phase 1 — additive (old + new + admin in the Secret)

| Measure | Result |
|---|---|
| New key first accepted | **+12 s** after the Secret change |
| New key last refused | **+22 s** (= rollout complete) |
| Client impact for the new credential | ~10 s window, 38 of 189 probes refused |
| Old key | never refused — additive rotation is safe for the *old* credential only |

Half of an additive rotation is deployed for as long as the rollout takes: a
client handed the new key immediately gets 401s from the pods that have not
rolled yet. Tell the client to retry, or hand over the key after the rollout.

### Phase 2 — removal (revocation), rolling

| Measure | Result |
|---|---|
| Revoked key first refused | **+12 s** after the Secret change |
| Revoked key **last accepted** | **+21 s** after the Secret change |
| Requests with the revoked key that still succeeded | **45 of 195 probes (~23 %)** |
| Rollout complete | +23 s |

**Revocation is not atomic.** Until the last old pod is gone, the Service keeps
routing to it and the revoked credential keeps working. On this cluster that was
21 s; a production rollout with image pulls and slower probes is longer.

### Phase 2b — the same revocation with `strategy: Recreate`

| Measure | Result |
|---|---|
| Revoked key last accepted | **+0 s** (the apply second) |
| Total unavailability | **~7 s** (4 consecutive connection failures) |
| First 401 from a new pod | +10 s |

So the incident choice is explicit: **rolling = 21 s of valid revoked key and no
outage; Recreate = 0 s of revoked key and ~7 s of outage.** For a compromised
credential prefer `Recreate` (or scale the old ReplicaSet to zero) and accept the
gap; for a scheduled rotation use rolling and two phases.

### Phase 3 — the runtime API's refusal, measured

With the auth routes mounted (`jwt_enabled=true`) and **no** keys file, in
production: supplied key → `409 not_persisted`; generated key →
`409 not_persisted`; `/v1/info` reports `runtime_rotation_durable: false`. The
runbook's claim is honest, and the posture tells an operator which mode a pod is
in *before* they try.

### Phase 4 — a durable SHARED key store does NOT make rotation fleet-wide

`NEURALGUARD_AUTH_KEYS_FILE` on a shared volume (RWX stand-in on a single-node
cluster), same two pods:

| Step | Pod A | Pod B |
|---|---|---|
| Before | `key_count 2`, `key_set_id 42a406a95c5d` | identical |
| `POST /v1/auth/keys/rotate` **on A** | `200 persisted:true`, count **3**, id `c3e4a2722f6e`, new key **accepted** | count **2**, id unchanged, new key **refused (401)** — while B can read the same file |
| After a rolling restart | count 3, id `c3e4a2722f6e`, new key accepted | identical |

The store is durable; the **reload happens at boot** (or on that pod's own
rotation call). A shared file therefore buys durability, not propagation — and
`file + restart` is equivalent to `env + restart` in effect. **Runtime rotation
is per process by construction in this build**, which is exactly why the
documented multi-replica path is env/redeploy.

### Defect this drill found (fixed): a failed write left the key live

With a keys volume the container user could not write, `POST /v1/auth/keys/rotate`
answered **`500 internal_error`** — and the new key was **live on that pod only**
(`key_count` 2 → 3, a new `key_set_id`), persisted nowhere, lost on restart. A
split-brain credential reported as a failure, which is the one outcome a
fail-closed control must never produce.

Fixed in this tree: `RuntimeKeyStore.add`/`remove` **persist first, then apply**
(an unwritable store leaves the live set untouched), and the route answers
`500 {"error": "key_store_write_failed"}` — a 500 that is *true*, because the
rotation did not happen. Contracts in `tests/unit/test_rotation_persist_failure.py`
written and run red first.

### Checking the posture after any rotation

```bash
for pod in $(kubectl get pods -n neuralguard -l app.kubernetes.io/name=neuralguard \
             -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}'); do
  kubectl exec -n neuralguard "$pod" -- \
    curl -s -H "Authorization: Bearer $ADMIN_KEY" localhost:8000/v1/info \
    | tr ',' '\n' | grep -E 'key_count|key_set_id' | tr '\n' ' '; echo " <- $pod"
done
```

Every pod must report the same `key_set_id`. Differing ids are not a cosmetic
detail: they are the list of pods still accepting the credential you think you
revoked. The fingerprint is built from `sha256(key)` plus tenant/agent bindings
and never contains key material.

## Runtime rotation API (P2-4)

Static env + restart is not the only path anymore. When the auth routes are
mounted (`NEURALGUARD_AUTH_JWT_ENABLED` and/or `NEURALGUARD_AUTH_KEYS_FILE`
set):

- `POST /v1/auth/keys/rotate` (admin-tenant only) issues a new key
  (generated or supplied), optionally revokes the caller's key, updates the
  middleware's live key map immediately, and — when
  `NEURALGUARD_AUTH_KEYS_FILE` is set — persists it durably (atomic 0600
  write, reloaded by every worker at boot).
- Runtime-only rotation (no keys file) is REFUSED in production — a rotation
  that evaporates on restart is a footgun, not a feature.
- Multi-worker note (measured 2026-11-17, see "Drilled" above): a rotated key
  is visible **per process** until each worker rotates or reloads, and a durable
  *shared* keys file does not change that — the reload happens at boot. For
  fleet-wide rotation use the env/redeploy path above; the endpoint is for
  single-appliance operation.
- `POST /v1/auth/token` exchanges a valid credential for a short-lived JWT
  bound to the caller's tenant — the static key can then stay in the
  deployment config only.

## Rotating the Postgres audit password (zero downtime)

Postgres supports two simultaneous passwords via `ALTER USER ... PASSWORD`
plus a `VALID UNTIL` window, but the simplest zero-downtime path is:

1. **Create a new role/password** (or `ALTER USER neuralguard PASSWORD
   'new'`). Both old and new passwords are accepted by Postgres for the
   same role during the window in which connections either have the old
   or new password cached.
2. **Update `NEURALGUARD_AUDIT_POSTGRES_URL`** with the new password and
   rolling-restart workers. New connections use the new password.
3. **Wait** for `pool_recycle` (default 1800s) to drain old connections,
   or restart all workers to force it.
4. **Verify** audit writes succeed: `curl -H "X-API-Key: ..." http://host/v1/metrics | grep audit_persist_failures`
   stays at 0.
5. **Revoke** the old password (`ALTER USER neuralguard PASSWORD 'new'`
   already superseded it; no separate revoke needed in Postgres).

For a managed DB (RDS/CloudSQL), use the managed credential rotation of
the secret manager and the proxy sidecar pattern so the app picks up the
new password without restart.

## Rotating the audit signing key (Ed25519, P2-10)

When `NEURALGUARD_AUDIT_SIGNING_KEY` is set, every audit event's chain hash
is signed and the signature rides on the event (`event_sig`). Signatures
verify against the public key of the key that made them, so rotation works
by epoch:

1. Generate the new keypair: `neuralguard audit-keygen` (prints the seed for
   `NEURALGUARD_AUDIT_SIGNING_KEY` and the derived pubkey for
   `audit-verify --pubkey`).
2. Dual-epoch window: deploy workers with the new seed; events written
   during the window may carry either epoch's signature.
3. Verify per epoch: run `neuralguard audit-verify --pubkey <old>` over the
   old window's files (or `--pg-url` + `--pubkey <old>` for Postgres-audit
   deployments), then `--pubkey <new>` over the new window. A file verified
   with the wrong epoch's pubkey reports BROKEN — that is the signature
   doing its job, not a verification failure.
4. Retire the old seed to the secret manager (marked retired).

On suspected key compromise: rotate immediately (new epoch), preserve the
old pubkey + audit files, and treat the pre-rotation window as
forensically suspect.

## Verification after any rotation

```bash
# 1. App boots and is ready:
curl -s -H "Authorization: Bearer <new>" http://host/v1/ready | jq .ready   # true

# 2. Auth still works with the new key, 401 with the old:
#    (the middleware accepts `Authorization: Bearer <key>` — shown here —
#    or `X-API-Key: <key>`; both are equivalent.)
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer <new>" http://host/v1/info   # 200
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer <old>" http://host/v1/info   # 401

# 3. Audit chain still verifies (P1-4/P2-10):
neuralguard audit-verify ./audit_logs                  # JSONL: per-worker chains, VALID/BROKEN
neuralguard audit-verify --pg-url "$PG_DSN"            # Postgres-audit deployments
neuralguard audit-verify ./audit_logs --pubkey "$PUB"  # + Ed25519 signatures when signing is on
```

## What is NOT covered here

- OIDC / RS256 issuance (P2-4 residual): tokens are HS256 issued by
  NeuralGuard itself; external IdP discovery (JWKS fetch) is a documented
  follow-up, not shipped.
- Refresh tokens (P2-4 residual): JWTs are short-lived and re-issued by
  exchanging a valid credential; there is no refresh-token flow yet.
- Vault/SOPS integration (P2-4 residual): the secret manager is referenced
  by these runbooks but not integrated in code.