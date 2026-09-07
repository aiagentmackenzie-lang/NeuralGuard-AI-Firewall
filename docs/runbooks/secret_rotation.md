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
- Multi-worker note: a rotated key is visible per-process until each worker
  rotates/reloads; for fleet-wide rotation use the env/redeploy path above
  (the runbook position: the endpoint is for single-appliance operation).
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