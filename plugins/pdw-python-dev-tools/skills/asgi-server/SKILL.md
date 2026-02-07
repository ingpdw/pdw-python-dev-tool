---
name: asgi-server
description: >
  Guides the agent through running and configuring ASGI servers (Uvicorn, Granian, Hypercorn)
  for Python web applications. Triggered when users say "run a FastAPI app",
  "configure uvicorn", "set up ASGI server", "deploy with uvicorn", "configure workers",
  "set up SSL/TLS", "run development server", "configure hot reload", or mention
  ASGI server, production deployment, server configuration, uvicorn, granian, or hypercorn.
version: 1.0.0
---

# Uvicorn ASGI Server

## Overview

Uvicorn is a lightning-fast ASGI server implementation for Python. Built on top of `uvloop` and `httptools`, it provides a minimal, high-performance interface for running asynchronous web frameworks such as FastAPI, Starlette, and other ASGI-compatible applications. Uvicorn supports HTTP/1.1, HTTP/2 (via the `[standard]` extras), WebSockets, and lifespan protocol events out of the box.

Key characteristics:
- High-throughput async I/O powered by `uvloop` (when installed)
- First-class support for FastAPI and Starlette applications
- Built-in hot reload for development workflows
- Multiprocess worker mode for production deployments
- SSL/TLS termination support
- WebSocket and HTTP/2 protocol support
- ASGI lifespan event handling for startup/shutdown logic

Install with standard extras for production-grade performance:

```bash
uv add "uvicorn[standard]"
```

The `[standard]` extras bundle includes `uvloop`, `httptools`, `watchfiles` (for reload), and `websockets`.

## Development Mode

Run a FastAPI application in development with automatic code reloading:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Integrate with uv for virtual environment management:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Restrict reload watching to specific directories to avoid unnecessary restarts:

```bash
uvicorn app.main:app --reload --reload-dir src --reload-dir templates
```

The `--reload` flag uses `watchfiles` (if installed via `[standard]`) for efficient filesystem monitoring. Avoid using `--reload` in production -- it is intended only for development.

## CLI Options Reference

| Option                    | Default       | Description                                                    |
|---------------------------|---------------|----------------------------------------------------------------|
| `--host`                  | `127.0.0.1`  | Bind address. Use `0.0.0.0` to accept external connections.   |
| `--port`                  | `8000`        | Port number to listen on.                                      |
| `--reload`                | off           | Enable auto-reload on code changes.                            |
| `--reload-dir`            | `.`           | Directory to watch for changes (repeatable).                   |
| `--workers`               | `1`           | Number of worker processes (multiprocess mode).                |
| `--log-level`             | `info`        | Log verbosity: `critical`, `error`, `warning`, `info`, `debug`, `trace`. |
| `--access-log`            | on            | Enable access log. Use `--no-access-log` to disable.          |
| `--proxy-headers`         | off           | Trust `X-Forwarded-For` and `X-Forwarded-Proto` headers.      |
| `--forwarded-allow-ips`   | `127.0.0.1`  | Comma-separated IPs trusted for proxy headers. Use `*` for all. |
| `--ssl-keyfile`           | none          | Path to SSL private key file.                                  |
| `--ssl-certfile`          | none          | Path to SSL certificate file.                                  |

## Programmatic Usage

Run Uvicorn programmatically within a Python script. This pattern is useful for debugging, testing, and IDE-based launch configurations.

```python
import uvicorn

from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
```

Pass the application as an import string (`"app.main:app"`) when using `reload=True` so that uvicorn can re-import the module on changes. Pass the application object directly only when reload is disabled.

### Programmatic Configuration Object

For more control, use `uvicorn.Config` and `uvicorn.Server`:

```python
import uvicorn

config = uvicorn.Config(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    log_level="info",
    access_log=True,
    workers=1,
)
server = uvicorn.Server(config)
server.run()
```

This approach is useful when embedding uvicorn inside a larger application or when fine-grained control over server lifecycle is required.

## Factory Pattern

Use the `--factory` flag when the application is created by a factory function rather than being a module-level object:

```bash
uvicorn app.main:create_app --factory
```

The factory function must be a zero-argument callable that returns an ASGI application:

```python
# app/main.py
from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="My API")

    # Register routes, middleware, event handlers
    from app.api import router
    app.include_router(router)

    return app
```

The factory pattern is preferred for larger applications because it avoids import side effects and enables cleaner testing and configuration.

## Worker Configuration

For production deployments on multi-core machines, use the `--workers` flag to spawn multiple processes:

```bash
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

Each worker runs its own event loop and handles requests independently. The `--workers` flag cannot be combined with `--reload`.

Calculate the optimal worker count based on available CPU cores:

```
workers = (2 * CPU_CORES) + 1
```

For I/O-bound applications (typical for FastAPI), this formula provides a good starting point. Tune based on actual load testing results.

For more robust multiprocess deployments, consider using Gunicorn with Uvicorn workers. Consult the [deployment reference](references/deployment.md) for Gunicorn integration patterns.

## Logging Configuration

### Basic Log Levels

Set the log level via CLI or programmatic configuration:

```bash
uvicorn app.main:app --log-level debug
```

Available levels in increasing verbosity: `critical`, `error`, `warning`, `info`, `debug`, `trace`.

### Access Log Control

Disable access logs in production when a reverse proxy already handles request logging:

```bash
uvicorn app.main:app --no-access-log
```

### Custom Log Configuration

Provide a custom logging configuration dictionary for structured or JSON logging:

```python
import uvicorn

log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "fmt": '%(asctime)s %(levelname)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_config=log_config)
```

Pass a YAML or JSON file path to `--log-config` on the CLI for externalized log configuration.

## Lifespan Handling

Uvicorn supports the ASGI lifespan protocol, which allows applications to run initialization and teardown logic on server start and stop.

### Lifespan Modes

| Mode   | Behavior                                                        |
|--------|-----------------------------------------------------------------|
| `on`   | Always send lifespan events. Fail if the app does not handle them. |
| `off`  | Never send lifespan events.                                     |
| `auto` | Send lifespan events if the app supports them; ignore otherwise. Default. |

Set via CLI:

```bash
uvicorn app.main:app --lifespan auto
```

### FastAPI Lifespan Context Manager

Define startup and shutdown logic with the FastAPI lifespan context manager:

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: initialize resources
    app.state.db_pool = await create_db_pool()
    app.state.redis = await create_redis_client()
    yield
    # Shutdown: release resources
    await app.state.db_pool.close()
    await app.state.redis.close()


app = FastAPI(lifespan=lifespan)
```

Use lifespan for database connection pools, cache clients, ML model loading, and other resources that require explicit initialization and cleanup.

## HTTP/2 and WebSocket Support

### HTTP/2

Enable HTTP/2 support by installing the `[standard]` extras and providing SSL certificates:

```bash
uvicorn app.main:app --ssl-keyfile key.pem --ssl-certfile cert.pem --http h2
```

HTTP/2 requires TLS. For local development, generate self-signed certificates with `mkcert` or `openssl`.

### WebSockets

WebSocket support is included by default when `websockets` or `wsproto` is installed (bundled in `[standard]`). Select the implementation explicitly:

```bash
uvicorn app.main:app --ws websockets
# or
uvicorn app.main:app --ws wsproto
```

No additional configuration is required -- FastAPI `WebSocket` endpoints work automatically.

### Protocol Selection

Select the HTTP implementation explicitly when needed:

```bash
# Use httptools (default with [standard], fastest)
uvicorn app.main:app --http httptools

# Use h11 (pure Python, no C dependencies)
uvicorn app.main:app --http h11

# Use h2 for HTTP/2
uvicorn app.main:app --http h2 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

The `httptools` implementation is recommended for production due to its superior throughput. The `h11` implementation is useful when C extensions are unavailable or when debugging protocol-level issues.

## SSL/TLS Configuration

Terminate TLS directly at Uvicorn when not running behind a TLS-terminating reverse proxy:

```bash
uvicorn app.main:app \
    --ssl-keyfile /etc/ssl/private/server.key \
    --ssl-certfile /etc/ssl/certs/server.crt \
    --ssl-ca-certs /etc/ssl/certs/ca-bundle.crt \
    --host 0.0.0.0 \
    --port 443
```

For most production deployments, terminate TLS at the reverse proxy (Nginx, Caddy, or a cloud load balancer) and run Uvicorn on plain HTTP behind it. Consult the [deployment reference](references/deployment.md) for Nginx TLS termination patterns.

## Running Behind a Reverse Proxy

When Uvicorn runs behind Nginx, Caddy, or a cloud load balancer, enable proxy header forwarding to preserve client IP addresses and protocol information:

```bash
uvicorn app.main:app \
    --proxy-headers \
    --forwarded-allow-ips "127.0.0.1,10.0.0.0/8" \
    --host 0.0.0.0 \
    --port 8000
```

- `--proxy-headers` tells Uvicorn to trust `X-Forwarded-For` and `X-Forwarded-Proto` headers.
- `--forwarded-allow-ips` restricts which upstream IPs are trusted. Set to `*` only in fully trusted network environments.

Without these flags, `request.client.host` will always report the proxy's IP rather than the actual client IP.

## Resource Limits

Control connection and request limits to prevent resource exhaustion in production:

```bash
uvicorn app.main:app \
    --limit-concurrency 100 \
    --limit-max-requests 10000 \
    --timeout-keep-alive 5 \
    --host 0.0.0.0 --port 8000
```

| Option                    | Default       | Description                                                    |
|---------------------------|---------------|----------------------------------------------------------------|
| `--limit-concurrency`     | none          | Maximum number of concurrent connections before rejecting new ones. |
| `--limit-max-requests`    | none          | Maximum requests per worker before restarting (guards against memory leaks). |
| `--timeout-keep-alive`    | `5`           | Seconds to wait for a new request on a keep-alive connection.  |
| `--timeout-notify`        | `30`          | Seconds to wait for graceful shutdown of a worker.             |
| `--backlog`               | `2048`        | Maximum length of the pending connections queue.               |

Set `--limit-max-requests` in production to guard against gradual memory leaks. Combine with `--workers` so that restarting one worker does not drop all traffic.

---

## Common Patterns

### Integration with uv

Run Uvicorn through uv to ensure the correct virtual environment and dependencies:

```bash
# Development
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uv run uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Environment-Based Configuration

Use environment variables or pydantic-settings to drive server configuration:

```python
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
        workers=settings.server_workers,
        log_level=settings.log_level,
    )
```

### Health Check Endpoint

Pair Uvicorn with a health check endpoint for load balancer and orchestrator integration:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Binding to a Unix Socket

Bind Uvicorn to a Unix domain socket instead of a TCP port for communication with a local reverse proxy. This eliminates TCP overhead:

```bash
uvicorn app.main:app --uds /tmp/uvicorn.sock
```

Ensure the reverse proxy (Nginx or Caddy) is configured to proxy to the same socket path. Unix sockets are not accessible over the network and are only suitable when the proxy runs on the same host.

### Restricting Reload File Types

During development, limit the file extensions that trigger a reload to avoid restarting on irrelevant file changes:

```bash
uvicorn app.main:app --reload --reload-include "*.py" --reload-include "*.yaml" --reload-exclude "test_*"
```

The `--reload-include` and `--reload-exclude` options accept glob patterns and can be repeated. This is particularly useful in projects with large static asset directories or generated files that should not trigger server restarts.

## Cross-References

- For Docker deployment, consult the `docker-build` skill.
- For FastAPI app configuration, consult the `fastapi` skill.
- For production deployment patterns with Gunicorn and Nginx, consult the [deployment reference](references/deployment.md).
- For dependency management with uv, consult the `uv` skill.
