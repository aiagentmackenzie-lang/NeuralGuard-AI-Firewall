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
- The image used here is **built locally**, not published. A real deployment
  must build/push to your registry and sign it (`docs/runbooks/artifact_signing.md` §4).

**Not drilled (and still not claimed):** prometheus-adapter custom-metric
scaling (template remains commented in `hpa.yaml` — no in-cluster Prometheus
was installed), ingress/TLS (deliberately out of scope), and multi-replica
runtime key rotation (production refuses it; env/redeploy is the documented
path).
