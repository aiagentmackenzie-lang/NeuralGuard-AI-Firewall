# Secret Rotation Runbook (P0-3)

Two secrets are in active use: the **API keys** (`NEURALGUARD_AUTH_API_KEYS`,
which authenticate clients and bind them to tenants) and the **Postgres
password** (`POSTGRES_PASSWORD` / the password embedded in
`NEURALGUARD_AUDIT_POSTGRES_URL`, when `audit.backend=postgres`). Rotate
both without downtime using a dual-key window.

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

## Verification after any rotation

```bash
# 1. App boots and is ready:
curl -s -H "X-API-Key: <new>" http://host/v1/ready | jq .ready   # true

# 2. Auth still works with the new key, 401 with the old:
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: <new>" http://host/v1/info   # 200
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: <old>" http://host/v1/info   # 401

# 3. Audit chain still verifies (P1-4):
python -m neuralguard.logging.chain  # or replay today's JSONL through verify_chain
```

## What is NOT covered here

- API-key issuance API (P2-4): there is no rotation endpoint; rotation is
  an env + restart today. Enterprise deploys should front NeuralGuard with
  an API gateway that owns key issuance.
- Key material for event signing (P2): audit events are hash-chained, not
  signed, so there is no signing key to rotate yet.