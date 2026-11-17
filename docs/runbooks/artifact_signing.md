# Runbook — Artifact signing (cosign): SBOM sign + attest, image signing

> P2-5. NeuralGuard ships with a **signed supply chain surface**: the SBOM is
> signed and attested in CI (keyless cosign), and release artifacts can be
> signed locally with a persistent key. This runbook covers all three flows
> and their verification stories.

## What exists today

| Flow | Where | Key | Status |
|:--|:--|:--|:--|
| SBOM blob signing (keyless) | CI job `sbom-sign` | ephemeral Fulcio cert (GitHub OIDC) | ✅ CI-verified — green on every push since 2026-09-05 |
| SBOM attestation (cyclonedx predicate) | CI job `sbom-sign` | same | ✅ same |
| Local artifact signing (key-based) | operator machine (this runbook) | `~/.cosign/<prefix>.key` | ✅ locally verified (sign → verify EXIT 0, cosign v3.1.3) |
| Container image signing | CI job `image` (`.github/workflows/image.yml`) | ephemeral Fulcio cert (GitHub OIDC) | ⏳ WIRED — the workflow exists and verifies its own signature + both attestations; the first successful publish is what proves the claim |

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

## 4. Container image — build, publish, sign (CI)

`.github/workflows/image.yml` owns the image. It runs on a `v*` tag (and on
demand) and does five things:

1. **Refuses a lying tag.** The tag must equal `pyproject.toml`'s version, or the
   job fails before anything is built — an image whose
   `org.opencontainers.image.version` label disagrees with its tag is worse
   than no image.
2. **Builds `linux/amd64` + `linux/arm64`** from the digest-pinned Dockerfile
   with `provenance: mode=max` and `sbom: true`, so the registry carries
   provenance and an SPDX SBOM attestation alongside the image.
3. **Pushes** `ghcr.io/<owner>/neuralguard:<version>` and `:sha-<short>`, plus
   `:latest` on a tag push only. `<owner>` is derived from the repository owner,
   so a fork publishes under its own namespace instead of colliding.
4. **Signs the digest** (never the tag) keyless, **attests the dependency SBOM**
   (CycloneDX, generated from the same extras the image installs) as a DSSE
   attestation in the registry, then verifies its own signature and that
   attestation with an identity regexp pinned to `.../.github/workflows/image.yml@`.
   A signature from any other workflow or repository fails the job.
5. **Checks the buildx attestations structurally.** `provenance: mode=max` and
   `sbom: true` attach SLSA provenance and an SPDX SBOM as *attestation
   manifests inside the image index* — and the OCI referrers API does not list
   those, so `cosign verify-attestation` cannot see them. The job asserts they
   are present by reading the index from the registry, instead of claiming a
   check that fails on a correct image (that is exactly how the first publish
   attempt went red).
6. **Prints the digest pin line** in the run summary.

A `smoke` job runs *before* anything is pushed: it builds the image for one
platform, loads it locally, starts it with `--network none` and health-checks it
from inside. That is a regression guard for a real defect the first publish
attempt found — the image ran `uv run`, which re-resolved dependencies at
container start, so it needed DNS and would have died in an airgapped cluster.
A signed image that cannot start without egress is not worth signing.

Cut the release tag only from a green `main`: the image is the artifact a
consumer actually runs, so the CI gate must already be green on the commit it
was built from.

Reproducing a build locally, if you ever need to:

```bash
docker build -t ghcr.io/<owner>/neuralguard:dev .
docker buildx imagetools inspect ghcr.io/<owner>/neuralguard:dev --format '{{.Manifest.Digest}}'
```

### Pin the digest in the manifest

A tag is mutable; the digest is the identity that was signed. After a
successful run, paste the line from the run summary into
`deploy/kubernetes/deployment.yaml`:

```yaml
image: ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:<digest>
```

### Consumer verification (outside CI)

There are three separate things to check, and they are not checked the same way:

| What | How | Notes |
|:--|:--|:--|
| Signature | `cosign verify` | covers the digest, identity = this workflow |
| Dependency SBOM | `cosign verify-attestation --type cyclonedx` | DSSE attestation created by the workflow |
| Buildx provenance + SPDX SBOM | read the image **index** from the registry | index-embedded; **not** in the referrers API, so cosign cannot verify them |

```bash
IMAGE=ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:<digest>
IDENTITY='^https://github.com/aiagentmackenzie-lang/NeuralGuard-AI-Firewall/\.github/workflows/image\.yml@'
ISS=https://token.actions.githubusercontent.com

cosign verify --certificate-identity-regexp "$IDENTITY" --certificate-oidc-issuer "$ISS" "$IMAGE"

cosign verify-attestation --type cyclonedx \
  --certificate-identity-regexp "$IDENTITY" --certificate-oidc-issuer "$ISS" "$IMAGE"

# provenance + SPDX SBOM live in the index, not the referrers API
TOKEN=$(curl -fsS "https://ghcr.io/token?scope=repository:aiagentmackenzie-lang/neuralguard:pull&service=ghcr.io" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -fsS -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/vnd.oci.image.index.v1+json' \
  "https://ghcr.io/v2/aiagentmackenzie-lang/neuralguard/manifests/${IMAGE##*@}" \
| python3 -c 'import json,sys;print([m["annotations"]["vnd.docker.reference.type"] for m in json.load(sys.stdin)["manifests"] if (m.get("annotations") or {}).get("vnd.docker.reference.type")])'
```

`cosign tree "$IMAGE"` lists the signature and the SBOM attestation — and will
*not* list the buildx ones, for the reason above.

To repair a release whose signature or attestation is missing, dispatch the
workflow with the `digest` input: it signs and attests an already-published
image instead of rebuilding, so the release's tag does not move.

Admission control (verify before pull) is a cluster-side policy (Kyverno
`verifyImages` / Sigstore policy-controller). It is **not wired**: none of the
manifests carry a policy, and no cluster installs one. A cluster drill that
claims signed-image admission must show the refusal, not just the policy file.

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