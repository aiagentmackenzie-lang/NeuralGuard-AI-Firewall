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

# Install dependencies
RUN uv sync --no-dev --frozen

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