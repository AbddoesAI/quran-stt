# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps only (kept separate to exploit layer cache)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml requirements.txt ./
COPY src/ src/

# Install the package into a prefix directory for clean copy
RUN pip install --no-cache-dir --prefix=/install -e .

# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed package from builder
COPY --from=builder /install /usr/local

# Copy source and data directory
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser data/ data/

# Writable directories for outputs and cache
RUN mkdir -p /app/cache /app/logs /app/output && \
    chown -R appuser:appuser /app

USER appuser

# Environment defaults (override at runtime)
ENV STT_DEVICE=cpu \
    STT_COMPUTE_TYPE=int8 \
    STT_MODEL=large-v3 \
    STT_PRIMARY_LANGUAGE=ur \
    HADITH_CACHE_PATH=/app/cache/hadith_cache

ENTRYPOINT ["quran-stt"]
CMD ["--help"]
