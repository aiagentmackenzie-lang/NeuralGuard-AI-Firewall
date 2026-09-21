# Security Policy

NeuralGuard is a **defensive** middleware firewall for LLM APIs and agentic
pipelines. This policy covers two distinct concerns:

1. **Reporting vulnerabilities in NeuralGuard itself**.
2. **Responsible use of NeuralGuard as a security control.**

## Reporting vulnerabilities in NeuralGuard

If you find a security issue in NeuralGuard itself — for example a way to
bypass a detection layer, tamper with the audit chain, bypass tenant
binding or authentication, exfiltrate prompts, or crash the pipeline —
please report it responsibly:

- **Do not** open a public GitHub issue for security reports.
- Email: **raphael@mobiussec.com** with `[NeuralGuard security]` in the subject.
- Include a clear description, reproduction steps, and your assessment of impact.
- Please allow a reasonable window (default 90 days) before public disclosure.

## Scope

**In scope:** vulnerabilities in NeuralGuard's code — the API and middleware
stack, scanners, auth/tenant binding, the audit chain and its verification,
the appliance proxy, the MCP gateway, SIEM routing, its Docker/Kubernetes
deployment posture, and its dependency configuration.

**Out of scope:** the behavior of LLMs/agents behind the firewall (that is
the tool's purpose), attacks that require control of the host, and issues
fixed by upgrading supported dependency versions.

## Responsible use

NeuralGuard is defensive middleware, but it is still security software that
makes allow/block decisions about other people's traffic:

- Run it **only** on infrastructure you own or are authorized to operate.
- Do not point the appliance proxy at upstreams you are not authorized to use.
- Audit logging captures prompt-derived evidence — deploy in line with your
  jurisdiction's data-protection obligations, and use
  `NEURALGUARD_AUDIT_TOKENIZE_PII` when prompts may carry personal data.

## Supply-chain posture

NeuralGuard's own supply-chain posture, enforced in CI:

- Hash-pinned dependencies plus a blocking `pip-audit` gate
- CycloneDX SBOM generated, **signed and attested keylessly with cosign**
- GitHub Actions SHA-pinned; dependabot covers actions + pip
- ONNX model binaries are **never committed** — CI regenerates them from
  tracked sources and rebuilds the semantic corpus, so the artifacts you run
  are reproducible from source

## Known boundaries (not vulnerabilities — design limits)

Input-side detection has a provable ceiling (see the threat-model section of
the [README](README.md) and the citations there). Known, documented limits:

- Fast-than-the-model input filters (including this one) cannot universally
  distinguish adversarial from benign prompts
- Streaming requests are refused fail-closed (422) by design
- Cross-worker audit chains are per-worker authenticated; global write
  ordering requires a WORM sink
- HMAC canary tokens are detection signals, not forensics-grade watermarks

These are tracked in the [public roadmap](docs/ROADMAP.md).

## Supported versions

Only the latest minor release receives security fixes.