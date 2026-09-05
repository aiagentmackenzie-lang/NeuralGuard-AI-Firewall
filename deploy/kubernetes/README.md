# NeuralGuard on Kubernetes (P2-6)

Raw manifests (+ kustomize). **Validation status, honestly:**

- ✅ **Offline schema validation** — every manifest passes
  `kubeconform -strict -kubernetes-version 1.31.0` (run from repo root).
- ❌ **NOT cluster-tested.** No cluster exists on the dev machine. The
  following drill is **PENDING** and must not be claimed as done:
  - `kubectl apply -k deploy/kubernetes` end-to-end boot,
  - probes going healthy (`/v1/ready` dependency-aware readiness),
  - HPA actually scaling (needs metrics-server present),
  - prometheus-adapter custom-metric scaling (template in `hpa.yaml`),
  - production fail-fast behavior under the Secret-driven config,
  - key rotation path in multi-replica (env/redeploy — see below).

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
  `SEMANTIC_ENABLED=true`.
- **Judge**: point `JUDGE_OLLAMA_URL` at an Ollama you operate; the egress
  gate (`NEURALGUARD_SCANNER_JUDGE_ALLOW_EGRESS`) applies as everywhere else.
- **Ingress/TLS**: deliberately not included — TLS termination is
  deployment-specific (see `docs/runbooks/tls_termination.md`). The Service
  is ClusterIP; never expose it un TLS-fronted.
- **Image signing**: when you push the image, sign it —
  `docs/runbooks/artifact_signing.md` §4 (registry signing, ops step).

## Local validation (what was actually run)

```bash
kubeconform -strict -summary -kubernetes-version 1.31.0 \
  $(find deploy/kubernetes -name '*.yaml' ! -name 'kustomization.yaml')
# → 10 resources valid, exit 0 (kustomization.yaml is not a resource — excluded)
```