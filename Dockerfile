FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uv", "run", "neuralguard", "serve"]