# syntax=docker/dockerfile:1

FROM rust:1.88-slim-bookworm AS builder

WORKDIR /build/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/src ./src

RUN cargo build --release --locked --bins

FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="Socartes Rust Backend" \
      org.opencontainers.image.description="Rust backend and CLI compatibility binaries for Socartes." \
      org.opencontainers.image.source="https://github.com/kkkayye/Socartes-Rust" \
      org.opencontainers.image.licenses="MIT"

ENV PORT=8000 \
    RUST_LOG=info \
    SOCARTES_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/data

COPY --from=builder /build/backend/target/release/socartes-backend /usr/local/bin/socartes-backend
COPY --from=builder /build/backend/target/release/socartes /usr/local/bin/socartes
COPY --from=builder /build/backend/target/release/socartes-cli /usr/local/bin/socartes-cli
COPY --from=builder /build/backend/target/release/socartes_cli /usr/local/bin/socartes_cli
COPY --from=builder /build/backend/target/release/deeptutor /usr/local/bin/deeptutor
COPY --from=builder /build/backend/target/release/deeptutor-cli /usr/local/bin/deeptutor-cli
COPY --from=builder /build/backend/target/release/deeptutor_cli /usr/local/bin/deeptutor_cli

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null || exit 1

CMD ["socartes-backend"]
