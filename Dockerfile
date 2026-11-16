# Base images are pinned by digest, not by tag: a tag is mutable, and a signed
# image that silently rebuilds on a different base is not the artifact that was
# signed. Dependabot's docker ecosystem bumps the base (see .github/dependabot.yml);
# re-resolve by hand with:
#   docker buildx imagetools inspect python:3.11-slim --format '{{.Manifest.Digest}}'
#   docker buildx imagetools inspect ghcr.io/astral-sh/uv:0.13.7 --format '{{.Manifest.Digest}}'
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS base

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
# Pinned to a version + digest. `uv:latest` made the dependency resolver a
# floating input in a supply chain that otherwise hash-pins everything.
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY README.md LICENSE* ./
COPY src/ src/
# NG-6: the guarded-FPR probe sets — the boot-time self-check measures the
# loaded corpus against these (fpr_guard_file_missing otherwise, and the
# SLO surface reports null). Small tracked JSONL files, runtime-required.
COPY benchmarks/ng_vs_ns/benign_corpus.jsonl benchmarks/ng_vs_ns/benign_corpus.jsonl
COPY corpus/benign_hard_negatives.jsonl corpus/benign_hard_negatives.jsonl

# Install dependencies. Extras must cover every backend the appliance
# profiles can enable: db (postgres audit), redis (AG session store +
# rate-limit backend), tenants (YAML tenant registry), metrics (Prometheus),
# semantic (ONNX/tokenizer runtime for the semantic layer — the models
# themselves stay OUT of the image (gitignored build artifacts); profiles
# mount models/ at /app/models to enable the layer, which degrades
# gracefully when the mount is absent).
RUN uv sync --no-dev --extra db --extra redis --extra tenants --extra metrics --extra semantic --frozen

# Production stage
FROM base AS production

# Create non-root user
RUN groupadd -r neuralguard && useradd -r -g neuralguard -d /app -s /sbin/nologin neuralguard

# Copy source
COPY --from=base /app /app

# Create audit log directory
RUN mkdir -p /data/audit && chown -R neuralguard:neuralguard /data /app

USER neuralguard

# `uv run` is a resolver-aware launcher: invoked bare it decides the environment
# is stale (the image is synced with --no-dev, whose default group is `dev`) and
# tries to install the missing packages AT CONTAINER START. Measured: an
# unmodified `docker run --network none` failed with "error resolving DNS" while
# pulling `pyarrow`, and with network it mutated the image at runtime. A
# security gateway must start in an airgapped, read-only-root cluster, so the
# entrypoint never resolves: the console script is called directly and any
# `uv run` an operator invokes is forced to use what is already installed.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["/app/.venv/bin/neuralguard", "serve"]
