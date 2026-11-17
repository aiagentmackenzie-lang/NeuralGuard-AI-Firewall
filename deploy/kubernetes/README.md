# NeuralGuard on Kubernetes (P2-6)

Raw manifests (+ kustomize). **Validation status, honestly:**

- ✅ **Offline schema validation** — every manifest passes
  `kubeconform -strict -kubernetes-version 1.31.0` (run from repo root).
- ✅ **Cluster drill PASSED 2026-11-12** (kind v0.33.0 / k8s v1.37.0 on
  colima): apply, 2/2 Ready, probes, allow/block verdicts, signed postgres
  audit rows, HPA 2→4 on CPU load and back down, and dependency-aware
  readiness under a redis outage. Seven drill-found defects were fixed in
  this tree — evidence and the full list are in
  [Cluster drill](#cluster-drill--2026-11-12-pass-evidence-below).
- ✅ **Re-drilled 2026-11-17 against the PUBLISHED image**
  (`ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:8984cfdf…`, pulled from
  GHCR by the node, nothing loaded locally — containerd log in
  [that drill](#cluster-drill--2026-11-17-re-run-against-the-published-digest)).
  It found **two real deployment-path defects** that the 11-12 drill had
  missed: a boot-order race that disabled the Postgres audit path for the
  pod's lifetime, and a readiness probe that called that state `ok`. Both are
  fixed in this tree, proven in-cluster against a locally built image, and
  **shipped in v0.3.1**.
- ✅ **Re-drilled again 2026-11-17 against the v0.3.1 digest** — the artifact the
  manifest now pins (`ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:6d1c0022f146…`,
  pulled from GHCR by a fresh kind cluster). Full check table passed **and** the
  defect that motivated the release is gone: the late-Postgres sequence that used
  to need a `rollout restart` converges on its own (attempts 1–2 failed, attempt
  5 created the schema, `RESTARTS 0`). Evidence in
  [that drill](#cluster-drill--2026-11-17-re-run-against-v031-the-fix-in-the-artifact).
- ✅ **Two-pod HITL drill 2026-11-17** — the shared escalation store proven
  across **two pods behind a Service** (not just two processes): raise on A,
  approve on B, the held request resumes on A, the record is consumed,
  re-approval is `404`; plus the operator path through the Service (3 rounds)
  and a **memory-store negative control** that produced the exact
  `403 mcp_escalation_timeout` a misconfigured two-replica deployment ships.
  Evidence in [that drill](#two-pod-hitl-drill--2026-11-17). It also produced
  the per-pod posture (`/v1/info` → `mcp_gateway.hitl`) and turned an
  unlabelled 500 on a store outage into a named, audited `503`.
- ⚠️ **Not drilled:** prometheus-adapter custom-metric scaling (template in
  `hpa.yaml` stays commented), ingress/TLS (out of scope on purpose), and
  multi-replica runtime key rotation (production refuses it; env/redeploy is
  the documented path).

## Deploy

```bash
cp deploy/kubernetes/secret.example.yaml secret.yaml   # fill REAL values
kubectl apply -f secret.yaml -n neuralguard            # apply secrets FIRST
kubectl apply -k deploy/kubernetes
kubectl -n neuralguard rollout status deploy/neuralguard
```

## Design notes

- **Topology** mirrors `docker-compose.appliance.yml`: neuralguard + redis
  (rate-limit windows + Agent Guardian session state, F4) + postgres
  (durable audit events — the P1-4 chain and P2-10 signatures live here).
- **Replicas**: rate limits and AG sessions are SHARED state via redis, so
  scaling out is honest (unlike the per-pod memory backends).
- **Key rotation**: the runtime rotation endpoint refuses non-durable
  rotation in production; multi-replica rotation uses the documented
  env/redeploy path. A durable RWX keys volume is a possible follow-up —
  deliberately not shipped half-validated.
- **Semantic layer**: OFF in the stock manifests (the ONNX model is
  gitignored). Bake `models/` into a custom image or mount a PV and set
  `NEURALGUARD_SCANNER_SEMANTIC_ENABLED=true`.
- **Judge**: point `NEURALGUARD_SCANNER_JUDGE_OLLAMA_URL` at an Ollama you
  operate; the egress gate (`NEURALGUARD_SCANNER_JUDGE_ALLOW_EGRESS`) applies
  as everywhere else.
- **Ingress/TLS**: deliberately not included — TLS termination is
  deployment-specific (see `docs/runbooks/tls_termination.md`). The Service
  is ClusterIP; never expose it without TLS in front.
- **Image signing**: when you push the image, sign it —
  `docs/runbooks/artifact_signing.md` §4 (registry signing, ops step).

## Local validation (what was actually run)

```bash
kubeconform -strict -summary -kubernetes-version 1.31.0 \
  $(find deploy/kubernetes -name '*.yaml' ! -name 'kustomization.yaml')
# → 10 resources valid, exit 0 (kustomization.yaml is not a resource — excluded)
```

## Cluster drill — 2026-11-17 (re-run against v0.3.1, the fix in the artifact)

The first re-drill proved the *0.3.0* artifact and found two deployment-path
defects, which were then fixed and proven against a locally built image. This run
closes the loop: the manifests pin the **v0.3.1** digest and a fresh
`kind` cluster pulls it.

**Environment:** kind v0.33.0 / k8s v1.37.0 on colima, `darwin/arm64` host.

```
Pulled image "ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:6d1c0022f146…"
```

| Check | Result |
|---|---|
| Apply + rollout | 2/2 Running, `RESTARTS 0` both pods |
| `/v1/health` from inside | `"version":"0.3.1"`, `environment: production` |
| Probes | `/v1/ready` → 200 `healthy`, `audit_db:"ok"`, `redis:"ok"` |
| Auth posture | `/v1/info` unauthenticated → **401** |
| Verdicts | benign **200 `allow`** · injection **403** (`PI-D-001`) |
| Postgres audit | 2 rows, 2/2 signed, `global_seq` 1–2 |
| Audit chain | `audit-verify --pg-url` → per-worker **VALID** and **(global-order) VALID**, exit 0 |
| Redis outage | `/v1/ready` → **503** `redis:"fail"`; `/v1/health` stays **200**; restore → healthy |
| **Late Postgres (the fix)** | app scaled to 0, `postgres` Service deleted, app scaled to 2 → attempts 1–2 logged `db_init_attempt_failed`; Service restored at 22:23:31Z → **attempt 5 → `db_tables_created`**, **`RESTARTS 0`**, audit rows written afterwards |

The last row is the one that matters: on 0.3.0 this exact sequence left the
durable audit path dead until a `rollout restart`. On the published 0.3.1
digest it converges on its own, inside the retry budget, with nothing restarted.

## Cluster drill — 2026-11-17 (re-run against the published digest)

**Why this run exists.** The 11-12 drill proved the *manifests* while loading a
locally built image with `kind load docker-image`. After v0.3.0 published a
signed image and the manifest was pinned by digest, nothing had exercised the
artifact a stranger actually pulls — and the image content changed afterwards
(the entrypoint no longer shells out to `uv run`, so it must boot airgapped).

**Environment:** kind v0.33.0 / k8s v1.37.0 on colima (2 CPU / 4 GiB),
`darwin/arm64` host — so the node resolved and ran the **arm64** submanifest.

**The artifact was pulled, not loaded.** No `kind load` was run. containerd on
the node logged:

```
PullImage "ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:8984cfdf65b0…"
Pulled image "…@sha256:8984cfdf65b0…" with image id "sha256:37bc9736e0c3…",
  repo digest "…@sha256:8984cfdf65b0…", size "207191603" in 503.136434ms
```

(`kubectl get pod -o jsonpath='{..image}'` shows the pinned digest; the running
`/v1/health` reports `"version":"0.3.0"` — the artifact, not just the tree.)

### Verified on the published artifact

| Check | Result |
|---|---|
| Apply + rollout | `deployment "neuralguard" successfully rolled out`, 2/2 Ready |
| Probes | `/v1/ready` → 200 `{"ready":true,…,audit_db:"ok"}` |
| Version in the image | `/v1/health` → 200 `"version":"0.3.0"` |
| Auth posture | `/v1/info` without a key → **401** |
| Benign request | `200 {"verdict":"allow",…}` |
| Injection request | **403** `"rule_id":"PI-D-001"` (3 findings) |
| Postgres audit | 3 rows, 3/3 with `event_sig`, `global_seq` 1–3 |
| Global audit chain | `audit-verify --pg-url` → both **VALID**, exit 0; deleting the middle row → both **BROKEN**, exit 1 |
| HPA metrics | `cpu: 1%/70%, memory: 40%/75%`, `ScalingActive=True` |
| HPA scale-out/in | 2 → 8 (maxReplicas) → 2, from a CPU burner inside the pods |
| Redis outage | `/v1/ready` → **503** `redis:"fail"`; `/v1/health` stays 200; restore → 200 healthy |

**Environment limitation, not a manifest defect:** during scale-out only 3 of 8
pods became Ready — the 2-CPU kind node cannot host 8 × 250m requests plus the
burst. The HPA decision was correct (`SuccessfulRescale` 2→3→7→8) and it scaled
back to 2 on its own. A production node pool sized for `maxReplicas` does not
hit this.

### The two defects this run found (both fixed in this tree)

1. **Boot-order race disabled the durable audit path permanently.** The app
   pods started before the `postgres` Service had endpoints; the one-shot
   `create_all` in the lifespan failed with
   `[Errno -2] Name or service not known`, logged `db_init_failed` **once**, and
   never retried. Later connections succeeded, so every audit INSERT failed with
   `relation "audit_events" does not exist` — nine requests in, nothing was
   persisted, and only `kubectl rollout restart` recovered it. **Fix:**
   `neuralguard.db.init.initialise_postgres` retries with backoff
   (1, 2, 4, 8, 8 s; budget `NEURALGUARD_AUDIT_INIT_ATTEMPTS`, default 6), and
   reports the outcome instead of swallowing it.
2. **Readiness reported a healthy audit store it could not write to.**
   `_probe_db` ran `SELECT 1` — connectivity — so `/v1/ready` answered
   `audit_db: "ok"` and aggregate `status: "healthy"` while the schema was
   absent. **Fix:** the probe now also checks `to_regclass('audit_events')` and
   reports `fail` when the configured backend cannot persist (aggregate drops to
   `degraded`; the worker keeps serving because JSONL still records, which is the
   documented fallback).

### Proof the fixes work in-cluster (substitution, stated plainly)

This half was run against **`neuralguard:drill-fix-2026-11-17`, built locally
from the fix branch and loaded with `kind load docker-image`** — the published
digest does not contain the fix yet. That substitution is the exact mistake the
11-12 drill made, so it is named here rather than implied.

| Probe | Result |
|---|---|
| Late dependency | Deleted the `postgres` Service, scaled the app up: attempts 1–4 failed with the same DNS error (`attempt`/`attempts` logged each time), Service restored at +8 s, **attempt 5 → `db_tables_created`**, `audit_events` present, **`RESTARTS 0`** — no rollout was issued |
| Readiness with the schema dropped | `audit_db: "fail"`, aggregate `"degraded"` (previously `ok`/`healthy`) |
| Readiness after recovery | `audit_db: "ok"`, `"healthy"` |
| Audit chain on the fixed image | 3 rows signed, `audit-verify` both chains **VALID**, exit 0 |
| Redis outage on the fixed image | `/v1/ready` **503** `redis:"fail"`, `/v1/health` 200 |

**Not healed by design:** a table dropped *out of band* on a running pod needs a
restart — the app does not re-run schema creation mid-flight. Readiness now
says so, which is the difference that matters.

## Two-pod HITL drill — 2026-11-17

**Why this drill exists.** The shared HITL store (NG-15, `hitl_store=redis`) had
been proven across two **OS processes on one host** and never across two **pods
behind a Service** — which is how it will actually be deployed. The startup rule
cannot close the gap for you: it refuses the memory store when
`SERVER_WORKERS > 1`, and pods each run one worker. A two-replica deployment
with the per-worker store boots happily and every escalation times out.

**Environment:** kind v0.33.0 (k8s v1.37.0) on colima (2 CPU / 4 GiB), a fresh
cluster. Two `neuralguard` pods, `redis`, `postgres`, a stub MCP upstream, and a
tenant (`acme`) whose policy escalates `deploy_prod`. `hitl_store=redis` on
Redis **db 2** (db 0 = rate limit, db 1 = Agent Guardian), 20 s wait, TTL 300 s.
Requests pinned to a pod with `kubectl port-forward pod/…` for the deterministic
legs; the operator-path leg goes through the Service.

### Behavioural legs — against the **published 0.3.1 digest**

| Stage | Result |
|---|---|
| Raise on **pod A**, read pending from **pod B** | the same `escalation_id` is visible from B |
| Redis db 2 while pending | exactly `ng:hitl:rec:<id>` + `ng:hitl:index` |
| Approve from **pod B** | `200` |
| The request held on **pod A** | **`200` in 3.277 s**, body = the upstream's answer |
| Redis after use | `[]` — the record was consumed |
| Re-approve the same id | **`404`** — one approval, one call |
| Stub upstream log | `UPSTREAM_TOOLS_CALL {"name": "deploy_prod", "arguments": {"env": "prod"}}` |
| Audit chains | `MCP-HITL-RAISED` written by worker `e1c53f5c…` (pod A), `MCP-HITL-DECISION` by `dd212a71…` (pod B) — **different chains, one decision** |

### Operator path — through the Service (3 rounds)

Every request went to `http://neuralguard:8000` from an in-cluster client, so
kube-proxy chose the endpoint and nothing in the drill knew which pod answered.

| Round | Raise | Approve | Held request | Workers (from the audit trail) |
|---|---|---|---|---|
| 1 | `200` | `200` | `200` | raise `e1c53f5c…`, decision `dd212a71…` — **split across pods** |
| 2 | `200` | `200` | `200` | both on `e1c53f5c…` |
| 3 | `200` | `200` | `200` | both on `dd212a71…` |

Published as measured: a single Service round splits across pods **1 time in 3**
(kube-proxy is per-connection random), which is exactly why the deterministic
leg above pins each request to a pod instead of trusting luck.

### Negative control — the same drill with the mechanism removed

`hitl_store=memory`, still **2 pods**, still `environment=production`:

| Check | Result |
|---|---|
| Boot | **accepted** — 0 `Production startup refused` lines on either pod (each pod has `SERVER_WORKERS=1`, so the app cannot see the replicas) |
| Pending list from **pod B** | `{"escalations":[],"pending":0}` — pod B cannot see pod A's escalation |
| Pending list from **pod A** | the escalation is there (it was really raised) |
| The held request | **`403 mcp_escalation_timeout` after 20.010 s** |
| Stub upstream | nothing — the call never left |

That is the failure a two-replica deployment ships if it sets the wrong value,
and the reason the drill is a test rather than a demo.

### Posture leg — a per-pod answer instead of a manifest promise

`GET /v1/info` → `mcp_gateway.hitl` (new in this tree; the 0.3.1 digest has no
such field, so this leg ran on `neuralguard:ng15-tree`, built locally at commit
`44cd1b4` and `kind load`ed — a substitution, named rather than implied).

| Configuration | pod A | pod B |
|---|---|---|
| `hitl_store=redis` (live) | `store=redis, shared_across_workers=true, store_id=7f64949407d7` | identical `store_id` |
| `hitl_store=memory` (2 pods) | `shared_across_workers=false, store_id=null` | identical |
| `hitl_store=redis`, **unreachable** store | `store_id=d07bfbf55926` | — (different from the live id, which is the point of the fingerprint) |

Two pods with the same `store_id` provably share a store; a per-worker store
now says so out loud. The choice is still an operator's — but it is no longer
an invisible one.

### Store outage — the refusal named, and audited

Same tree image, `hitl_store=redis` pointed at a dead endpoint
(`127.0.0.1:6399`):

| Probe | Result |
|---|---|
| Gated call | `503` in **9 ms** — `{"error":"mcp_escalation_store_unavailable","rule_id":"MCP-HITL-STORE"}` |
| Approver polling | `503` same shape, and **no audit event** (polling must not be an amplifier) |
| Stub upstream | nothing forwarded |
| Audit trail | `MCP-HITL-STORE \| escalation_store_unavailable \| block` |

Before this change the same misconfiguration produced `500 internal_error` with
no rule id **and no audit event** — which is how the gap was found: see the
trap below.

### Two traps this drill found

1. **A strategic-merge patch inserts its `env` before the base container's.**
   Adding `NEURALGUARD_MCP_HITL_REDIS_URL: "redis://:$(REDIS_PASSWORD)@…"` in an
   overlay patch put it *ahead* of the `REDIS_PASSWORD` it interpolates, so the
   value stayed literal and `$(REDIS_PASSWORD)` became the Redis **username**
   (`invalid username-password pair`). The base manifest's rule — "the password
   ref must be defined EARLIER in this list" — holds for the base and is broken
   by a patch. Fix used here, and recommended for any overlay: put the whole
   URL in the Secret (`hitl-redis-url`) and reference it with `secretKeyRef`.
2. **A store outage used to be an unlabelled failure.** The trap above was
   invisible in the API response (a 500 with a correlation id) and absent from
   the audit trail, so the drill could only diagnose it from the pod's
   traceback. It is now a named 503 with its own rule id — see the CHANGELOG.

**Teardown:** cluster, stub, port-forwards and the drill overlay were removed;
no containers left running.

## Cluster drill — 2026-11-12 (PASS, evidence below)


**Environment:** kind v0.33.0 cluster (k8s v1.37.0) on colima (2 CPU / 4 GiB),
image built locally from this repo's Dockerfile and loaded with
`kind load docker-image`. Nothing was pushed to a registry.

**What was applied:** `kubectl apply -k deploy/kubernetes` + a generated
`secret.yaml` (all values `openssl rand`). Two replicas, redis, postgres, HPA.

### Verified

| Check | Result |
|---|---|
| Apply + rollout | `deployment "neuralguard" successfully rolled out` (2/2 Ready) |
| Probes | `/v1/ready` → `200 {"ready":true,"status":"healthy",...}` |
| Auth posture | `/v1/info` without a key → 401 (stays auth-gated) |
| Benign request | `200 {"verdict":"allow",...}` |
| Injection request | `403 {"verdict":"block","rule_id":"PI-D-001",...}` |
| Postgres audit | 2 rows in `audit_events`, both with `event_sig` (Ed25519) |
| Global audit chain | 3 chained rows (`global_seq` 7–9, each linked to its predecessor); deleting the middle row made `audit-verify --pg-url` report `(global-order)` **BROKEN** and exit 1 |
| HPA metrics | `cpu: 1%/70%, memory: 52%/75%`, `ScalingActive=True` |
| HPA scale-out | 2 → 4 replicas on CPU 118%/70% (`SuccessfulRescale`) |
| HPA scale-down | 4 → 3 → 2 on the 300 s stabilization window |
| Dependency-aware readiness | redis scaled to 0 → `/v1/ready` **503** `redis:"fail"`; `/v1/health` stays 200; redis restored → 200 healthy |

### Defects the drill found (all fixed in this tree)

Offline `kubeconform` validation cannot see any of these — each one was a
real boot/readiness failure on a live cluster:

1. **Redis crash-loop:** the image's entrypoint drops privileges with
   `setpriv`, which fails under `allowPrivilegeEscalation:false` + dropped
   caps (`setresuid failed`). Fixed by pinning the image's own uid/gid
   (999/1000) as the pod securityContext, disabling RDB (`--save ""`), and
   mounting an `emptyDir` at `/data` (readOnlyRootFilesystem).
2. **Postgres crash-loop:** same class — entrypoint `chown/chmod` as root
   with `CAP_CHOWN` dropped (`chmod: /var/run/postgresql: Operation not
   permitted`). Fixed with `runAsUser/runAsGroup/fsGroup: 70` (the image's
   uid) and a tmpfs at `/var/run/postgresql`.
3. **NeuralGuard would not start:** `runAsNonRoot: true` against the image's
   **named** `USER neuralguard` → `CreateContainerConfigError` ("image has
   non-numeric user"). Fixed with `runAsUser/runAsGroup: 999`.
4. **`NEURALGUARD_PORT` env collision:** Kubernetes injects every Service as
   `<NAME>_<PORT>`, so `neuralguard` Service produced
   `NEURALGUARD_PORT=tcp://10.96.114.2:8000` and pydantic refused to parse it
   as an int (boot crash). Fixed with `enableServiceLinks: false` on all
   three pod specs (also stops cluster topology leaking into the env).
5. **Production unknown-key refusal (the F5 gate working):** the Secret
   interpolation helpers were named `NEURALGUARD_POSTGRES_PASSWORD` /
   `NEURALGUARD_REDIS_PASSWORD` — not settings fields. Production boot
   refused, correctly. Renamed to `PG_PASSWORD` / `REDIS_PASSWORD`.
6. **Readiness probe 401:** `/v1/ready` is auth-gated by default, so the
   kubelet never saw it as Ready. Fixed with
   `NEURALGUARD_AUTH_PUBLIC_ENDPOINTS=["/v1/health","/v1/ready"]` (the
   route's own docstring documents this opt-in; `/v1/info` and `/v1/metrics`
   stay gated).
7. **Probes were rate-limited (app bug, not manifest):** with redis down the
   readiness probe got **429** instead of the dependency-aware 503 — and a
   429 on a probe makes the kubelet mark a healthy pod NotReady. Fixed in
   `middleware/ratelimit.py`: `/v1/health` and `/v1/ready` bypass the limiter
   (constant-work, unauthenticated, node-IP-shared requests). Regression
   tests in `tests/unit/test_ratelimit.py`.

### Environment limitations (not manifest defects)

- The kind node has 2 CPU / 4 GiB. During a `maxSurge: 1` rollout while the
  HPA is scaled out, the surge pod can be `Pending` (`Insufficient cpu`) —
  it schedules once the HPA returns toward `minReplicas`. A production node
  pool sized for `maxReplicas` does not hit this.
- The image used **in this drill** was built locally from that commit's
  Dockerfile and loaded with `kind load docker-image`. **Superseded
  2026-11-17:** the drill was re-run against the published digest
  (`ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:8984cfdf…`, pulled from
  GHCR by the node) — see the 2026-11-17 section above, which found two
  deployment-path defects this run could not see. The manifests are unchanged
  between the two runs; the artifact is not, which is the whole point.

**Not drilled (and still not claimed):** prometheus-adapter custom-metric
scaling (template remains commented in `hpa.yaml` — no in-cluster Prometheus
was installed), ingress/TLS (deliberately out of scope), and multi-replica
runtime key rotation (production refuses it; env/redeploy is the documented
path).
