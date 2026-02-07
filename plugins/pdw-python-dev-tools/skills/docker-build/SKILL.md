---
name: docker-build
description: >
  Guides building Docker images and composing containers for Python/FastAPI
  applications. Triggered when users ask to "create a Dockerfile",
  "dockerize a Python app", "optimize Docker image", "create docker-compose",
  "set up multi-stage build", "reduce Docker image size",
  "create development container", or "configure Docker for FastAPI". Covers
  Docker, Dockerfile, container, image build, docker-compose, and
  containerization best practices for production and development workflows.
version: 1.0.0
---

# Docker Build for Python/FastAPI Applications

## Overview

Containerize Python and FastAPI applications using Docker with
production-grade multi-stage builds, layer caching, security hardening, and
Docker Compose orchestration. All patterns target Python 3.12+, use `uv` for
dependency management, and follow the principle of minimal, reproducible
images.

Refer to the asset templates bundled with this skill for ready-to-use
starting points:

- `assets/Dockerfile.fastapi` -- production multi-stage Dockerfile
- `assets/Dockerfile.dev` -- development Dockerfile with hot reload
- `assets/docker-compose.yml` -- Compose stack with PostgreSQL and Redis
- `assets/.dockerignore` -- ignore rules for lean build contexts

---

## Base Image Selection

Choose the base image according to the deployment target and dependency
requirements.

### python:3.12-slim (recommended)

The default choice for most FastAPI projects. Based on Debian Bookworm with a
minimal package set. Binary wheels from PyPI install without issues, and
system libraries such as `libpq` can be added via `apt-get`.

```dockerfile
FROM python:3.12-slim AS base
```

### python:3.12-alpine

Smaller download size, but Alpine uses musl libc. Packages with C extensions
(e.g., `psycopg2`, `numpy`, `pandas`) often fail to install or require
building from source, negating the size advantage. Avoid unless the
dependency tree is pure Python.

### ghcr.io/astral-sh/uv:python3.12-bookworm-slim

Ships with `uv` pre-installed. Useful when the build should not fetch `uv`
at build time. The image is Debian-based and behaves like `python:3.12-slim`
otherwise.

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
```

---

## Multi-Stage Build Pattern

Separate dependency installation from the final runtime image to keep the
production image small and free of build tools.

### Stage 1 -- Builder

Install `uv`, copy only the dependency manifests, and resolve/install
dependencies into a virtual environment. This stage may contain compilers and
header packages that must not ship in production.

```dockerfile
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev
```

### Stage 2 -- Runtime

Start from a clean slim image, copy only the virtual environment and
application source from the builder, create a non-root user, and declare the
entrypoint.

```dockerfile
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## uv Integration in Docker

### Installing uv

The fastest method is a static binary copy from the official image:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv
```

Alternatively, install via pip (slower, but works on any base):

```dockerfile
RUN pip install --no-cache-dir uv
```

### Dependency Resolution

Always pass `--frozen` to `uv sync` inside Docker so that the lockfile is
used as-is without resolution. This guarantees reproducible builds.

```dockerfile
RUN uv sync --frozen --no-dev
```

For uv dependency management outside of Docker, consult the `uv` skill.

---

## Layer Ordering for Cache Efficiency

Docker caches each layer. When a layer changes, every subsequent layer is
rebuilt. Order instructions from least to most frequently changing:

1. Base image and system packages
2. Copy dependency manifests (`pyproject.toml`, `uv.lock`)
3. Install dependencies (`uv sync`)
4. Copy application source code
5. Any final build steps

```dockerfile
# Step 2 -- manifests change infrequently
COPY pyproject.toml uv.lock ./

# Step 3 -- re-runs only when manifests change
RUN uv sync --frozen --no-dev --no-install-project

# Step 4 -- changes on every code edit
COPY src/ src/
```

This ordering ensures that dependency installation is cached across most code
changes, drastically reducing rebuild times.

---

## Security Hardening

### Non-Root User

Never run the application as root inside the container. Create a dedicated
user and switch to it before `CMD`.

```dockerfile
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
```

For stricter security, use a numeric UID and no login shell:

```dockerfile
RUN adduser --system --uid 1001 --no-create-home appuser
USER 1001
```

### Minimal Packages

Do not install editors, debug tools, or documentation packages in the
production image. If build-time packages are necessary (e.g., `gcc`,
`libpq-dev`), install them only in the builder stage.

### .dockerignore

Exclude files that must not enter the build context: version control
metadata, local environment files, test suites, caches, and documentation.
See `assets/.dockerignore` for a comprehensive template.

### Pin Image Digests in CI

For reproducible CI builds, pin the base image to a digest:

```dockerfile
FROM python:3.12-slim@sha256:<digest> AS builder
```

---

## HEALTHCHECK Instruction

Define a health check so orchestrators (Docker Swarm, Compose, ECS) can
detect unresponsive containers and restart them automatically.

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"] || exit 1
```

If `curl` is available in the runtime image, prefer the simpler form:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

Ensure the FastAPI application exposes a lightweight `/health` endpoint that
returns HTTP 200.

---

## Environment Variables

Set these early in the Dockerfile to influence Python runtime behavior:

```dockerfile
# Send stdout/stderr straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Prevent .pyc file creation inside the container
ENV PYTHONDONTWRITEBYTECODE=1
```

Application-specific variables (database URL, secret keys) should be injected
at runtime via `docker run --env-file` or the Compose `env_file` directive,
never baked into the image.

---

## CMD with Exec Form

Always use the exec form (JSON array) for `CMD` so that `uvicorn` receives
signals directly from Docker and can shut down gracefully.

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Avoid the shell form (`CMD uvicorn ...`) because it wraps the process in
`/bin/sh -c`, which swallows `SIGTERM` and delays container stops.

For uvicorn production configuration (workers, timeouts, keep-alive),
consult the `uvicorn` skill.

---

## Docker Compose

Use Docker Compose to orchestrate the application alongside backing services
such as PostgreSQL and Redis. See `assets/docker-compose.yml` for a complete
template.

### Service Definitions

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
```

### Health Checks in Compose

Each service should declare a `healthcheck` so that `depends_on` with
`condition: service_healthy` works correctly.

```yaml
db:
  image: postgres:16-alpine
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U postgres"]
    interval: 10s
    timeout: 5s
    retries: 5
```

### Volumes

Persist database data across container recreations:

```yaml
volumes:
  postgres_data:
  redis_data:
```

### Networks

Isolate inter-service traffic on a dedicated bridge network:

```yaml
networks:
  backend:
    driver: bridge
```

---

## Development vs Production Dockerfiles

Maintain separate Dockerfiles for each environment.

### Production (`Dockerfile.fastapi`)

- Multi-stage build
- `--no-dev` dependencies only
- Source code copied into the image
- Non-root user, health check, exec-form CMD

### Development (`Dockerfile.dev`)

- Single stage for simplicity
- All dependencies including dev extras (`pytest`, `ruff`, etc.)
- Source code mounted via a bind volume -- do not `COPY` source
- `--reload` flag on uvicorn for hot reload

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

Override the production Compose file with a development variant:

```yaml
# docker-compose.override.yml
services:
  app:
    build:
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
```

---

## Build Arguments

Use `ARG` to parameterize the build without creating multiple Dockerfiles.

```dockerfile
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS builder

ARG APP_VERSION=0.0.0
LABEL org.opencontainers.image.version=${APP_VERSION}
```

Pass values at build time:

```bash
docker build --build-arg PYTHON_VERSION=3.13 --build-arg APP_VERSION=1.2.0 .
```

---

## Volume Mounts for Development

Bind-mount the project directory into the container so that file changes on
the host are immediately visible inside the container, enabling uvicorn's
`--reload` watcher.

```bash
docker run -v "$(pwd)":/app -p 8000:8000 myapp-dev
```

In Compose:

```yaml
services:
  app:
    volumes:
      - .:/app
```

Avoid bind-mounting over the `.venv` directory. If the virtual environment
lives inside the project tree, use an anonymous volume to shadow it:

```yaml
volumes:
  - .:/app
  - /app/.venv
```

---

## Image Size Reduction Checklist

1. Use `python:3.12-slim` instead of the full `python` image.
2. Apply multi-stage builds; keep compilers in the builder stage only.
3. Combine `RUN` commands where logical to reduce layer count.
4. Pass `--no-cache-dir` to pip or use `uv` (which never caches by default).
5. Remove apt lists after installing system packages:
   `RUN apt-get update && apt-get install -y --no-install-recommends pkg && rm -rf /var/lib/apt/lists/*`
6. Add a thorough `.dockerignore` to minimize the build context.
7. Compile bytecode at build time (`UV_COMPILE_BYTECODE=1`) and skip `.pyc`
   generation at runtime (`PYTHONDONTWRITEBYTECODE=1`).

---

## Multi-Architecture Builds

Build images for multiple platforms using `docker buildx`:

```bash
# Create a buildx builder
docker buildx create --name multiarch --use

# Build and push for amd64 and arm64
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:latest --push .
```

When using multi-arch builds, ensure all base images support the target platforms. Avoid architecture-specific binaries in `COPY` instructions.

---

## BuildKit Secrets

Mount secrets at build time without baking them into image layers:

```dockerfile
# syntax=docker/dockerfile:1
RUN --mount=type=secret,id=pip_index_url \
  PIP_INDEX_URL=$(cat /run/secrets/pip_index_url) \
  uv sync --frozen --no-dev
```

Pass the secret at build time:

```bash
docker build --secret id=pip_index_url,env=PIP_INDEX_URL .
```

Secrets are available only during the `RUN` instruction and never persist in the image history.

---

## Common Labels

Apply OCI-standard labels for image metadata:

```dockerfile
LABEL org.opencontainers.image.title="my-fastapi-app"
LABEL org.opencontainers.image.description="FastAPI application"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/org/repo"
```

---

## Cross-References

- For uvicorn production configuration (workers, timeouts, logging), consult
  the `uvicorn` skill.
- For uv dependency management (lockfiles, workspaces, scripts), consult the
  `uv` skill.
