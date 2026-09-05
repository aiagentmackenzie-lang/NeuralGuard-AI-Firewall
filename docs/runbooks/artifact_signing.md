# Runbook — Artifact signing (cosign): SBOM sign + attest, image signing

> P2-5. NeuralGuard ships with a **signed supply chain surface**: the SBOM is
> signed and attested in CI (keyless cosign), and release artifacts can be
> signed locally with a persistent key. This runbook covers all three flows
> and their verification stories.

## What exists today

| Flow | Where | Key | Status |
|:--|:--|:--|:--|
| SBOM blob signing (keyless) | CI job `sbom-sign` | ephemeral Fulcio cert (GitHub OIDC) | ✅ CI-verified once pushed (first green run pending push approval) |
| SBOM attestation (cyclonedx predicate) | CI job `sbom-sign` | same | ✅ same |
| Local artifact signing (key-based) | operator machine (this runbook) | `~/.cosign/<prefix>.key` | ✅ locally verified (sign → verify EXIT 0, cosign v3.1.3) |
| Container image signing | operator + registry | local key or keyless | 🟠 OPS STEP — requires a registry; not possible from CI without a push target (below) |

The public half of the local key pair is committed at `cosign/neuralguard-local.pub`
so anyone can verify locally-signed artifacts. **The private key is never
committed, never leaves the machine, never enters CI** — CI uses keyless
signing (identity = the workflow, verified via the OIDC issuer + identity
regexp, transparency log via Rekor).

## 1. Key generation (local, one-time)

```bash
mkdir -p ~/.cosign
COSIGN_PASSWORD="<passphrase>" cosign generate-key-pair \
  --output-key-prefix "$HOME/.cosign/neuralguard-local"
# → ~/.cosign/neuralguard-local.key (0600, NEVER commit)
# → ~/.cosign/neuralguard-local.pub (safe to commit/distribute)
cp "$HOME/.cosign/neuralguard-local.pub" cosign/neuralguard-local.pub
```

Key loss = can no longer produce signatures under this identity (rotate: new
prefix, commit the new `.pub`, document the epoch in the release notes). Key
theft = attacker can sign under this identity (rotate + note the compromise;
signatures carry timestamps so verification can pin epochs).

## 2. Local signing + verification (blob signing)

```bash
cd "$REPO_ROOT"
mkdir -p dist-sign

# Sign a blob (the SBOM here; the same command signs any release artifact)
COSIGN_PASSWORD="<passphrase>" cosign sign-blob \
  --key "$HOME/.cosign/neuralguard-local.key" \
  --bundle dist-sign/sbom.json.sig.bundle sbom.json

# Attest the blob with the SBOM as a cyclonedx predicate
# (the SBOM is CycloneDX — do NOT use --type spdxjson, that would be a false claim)
COSIGN_PASSWORD="<passphrase>" cosign attest-blob \
  --predicate sbom.json --type cyclonedx \
  --key "$HOME/.cosign/neuralguard-local.key" \
  --bundle dist-sign/sbom.json.attest.bundle sbom.json

# Verify (works anywhere with the committed pubkey + the bundles)
cosign verify-blob --key cosign/neuralguard-local.pub \
  --bundle dist-sign/sbom.json.sig.bundle sbom.json          # → "Verified OK", exit 0
cosign verify-blob --key cosign/neuralguard-local.pub \
  --bundle dist-sign/sbom.json.attest.bundle sbom.json       # → "Verified OK", exit 0
```

## 3. CI keyless flow (what `sbom-sign` does)

1. Regenerates `sbom.json` from the same commit (provenance = the commit the
   OIDC token identifies).
2. `cosign sign-blob --yes --bundle ... sbom.json` — signs under a Fulcio
   certificate issued against GitHub's OIDC token (10-min ephemeral key).
3. `cosign attest-blob --yes --predicate sbom.json --type cyclonedx --bundle ...`
   — DSSE attestation embedding the SBOM.
4. **Verifies both bundles with an identity regexp pinned to this repo + this
   workflow** (`^https://github.com/aiagentmackenzie-lang/NeuralGuard-AI-Firewall/\.github/workflows/ci\.yml@`)
   — a signature from any other repo/workflow fails the job.
5. Uploads both bundles as the `sbom-signing-bundles` workflow artifact.

Verify a CI signature outside CI:

```bash
cosign verify-blob \
  --bundle sbom.json.sig.bundle \
  --certificate-identity-regexp '^https://github.com/aiagentmackenzie-lang/NeuralGuard-AI-Firewall/\.github/workflows/ci\.yml@' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  sbom.json
```

## 4. Container image signing (OPS STEP — needs a registry)

Image signing attaches signatures to a registry (or attaches them to the image
itself in OCI 1.1+). CI cannot do this today because NeuralGuard's CI builds no
image and pushes to no registry — fabricating a signing claim here would be
vapor. When an image is published, the operator flow is:

```bash
# Build + push once (example: ghcr)
docker build -t ghcr.io/aiagentmackenzie-lang/neuralguard:vX.Y.Z .
docker push ghcr.io/aiagentmackenzie-lang/neuralguard:vX.Y.Z
DIGEST=$(docker buildx imagetools inspect ghcr.io/aiagentmackenzie-lang/neuralguard:vX.Y.Z \
  --format '{{json .Manifest.Digest}}' | tr -d '"')

# Key-based signing (recommended for an owned identity):
COSIGN_PASSWORD="<passphrase>" cosign sign --key "$HOME/.cosign/neuralguard-local.key" \
  --sign-registry-insecure-freshen 2>/dev/null || \
COSIGN_PASSWORD="<passphrase>" cosign sign --key "$HOME/.cosign/neuralguard-local.key" \
  "ghcr.io/aiagentmackenzie-lang/neuralguard@${DIGEST}"

# Attest the SBOM INTO the registry against the image:
COSIGN_PASSWORD="<passphrase>" cosign attest --key "$HOME/.cosign/neuralguard-local.key" \
  --predicate sbom.json --type cyclonedx "ghcr.io/aiagentmackenzie-lang/neuralguard@${DIGEST}"

# Verify (consumer side, pinned to our key):
cosign verify --key cosign/neuralguard-local.pub \
  "ghcr.io/aiagentmackenzie-lang/neuralguard@${DIGEST}"
cosign verify-attestation --key cosign/neuralguard-local.pub --type cyclonedx \
  "ghcr.io/aiagentmackenzie-lang/neuralguard@${DIGEST}"
```

Admission control (cluster-side verify before pull) is a K8s policy plugin
(Kyverno `verifyImages` / Sigstore policy-controller) — worth wiring when the
P2-6 manifests get a real cluster drill; note it in the drill checklist.
Keyless image signing follows the same pattern with `--key` omitted
(`COSIGN_FULCIO_*` defaults; verification via the identity regexp above).

## 5. Honest limits

- The local key is a demo/portfolio identity, generated on the operator
  machine — it is not a hardware-backed or org-scoped identity. For a real
  release surface, prefer KMS (the `azurekms://` / `gcpkms://` / `hashivault://`
  key refs work out of the box in cosign).
- Keyless signatures prove "CI of repo X produced this" — they do not prove
  the artifact is *good*. That is what the security scan (pip-audit) +
  coverage + boot-smoke gates are for.
- Bundles produced by sign-blob include the Rekor transparency-log proof where
  keyless is used; local key-based bundles are offline-verifiable without
  Rekor (no tlog claim).