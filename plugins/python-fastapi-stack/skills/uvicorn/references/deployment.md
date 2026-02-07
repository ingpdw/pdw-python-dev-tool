# Uvicorn Production Deployment Reference

## Server Architecture Comparison

Three primary deployment architectures exist for running ASGI applications in production. Choose based on operational requirements and infrastructure constraints.

### Uvicorn Standalone

Run Uvicorn directly with the `--workers` flag for multiprocess operation:

```bash
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

**Advantages:**
- Simplest deployment configuration
- Single dependency (uvicorn)
- Sufficient for containerized deployments where the orchestrator handles process supervision

**Disadvantages:**
- Limited process management capabilities
- No graceful worker restart on code changes
- No configurable worker recycling

**Best for:** Docker/Kubernetes deployments where the container runtime manages process lifecycle.

### Gunicorn + Uvicorn Workers

Use Gunicorn as a process manager with Uvicorn worker classes:

```bash
gunicorn app.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -
```

**Advantages:**
- Mature process management with automatic worker restart
- Graceful worker recycling via `--max-requests`
- Pre-fork model with proper signal handling
- Configurable worker timeout and graceful shutdown

**Disadvantages:**
- Additional dependency (gunicorn)
- Gunicorn does not support Windows
- Slightly more complex configuration

**Best for:** Traditional Linux server deployments, VM-based infrastructure, and any environment where robust process supervision is needed outside of a container orchestrator.

### Hypercorn

An alternative ASGI server supporting HTTP/1, HTTP/2, and HTTP/3 (QUIC):

```bash
hypercorn app.main:app --bind 0.0.0.0:8000 --workers 4
```

**Advantages:**
- Native HTTP/3 and QUIC support
- Trio async framework support in addition to asyncio
- Single-binary deployment

**Disadvantages:**
- Smaller community and ecosystem than uvicorn
- Fewer deployment guides and operational knowledge

**Best for:** Applications requiring HTTP/3 support or Trio-based async.

## Calculating Optimal Worker Count

The optimal number of worker processes depends on the workload profile and available resources.

### Formula

```
workers = (2 * CPU_CORES) + 1
```

For a 4-core machine: `(2 * 4) + 1 = 9` workers.

### Guidelines

- **I/O-bound workloads** (database queries, HTTP calls, file I/O): Use the formula as a starting point. Async frameworks like FastAPI already handle concurrency within each worker, so fewer workers may suffice.
- **CPU-bound workloads** (data processing, image manipulation): Match the worker count to the number of CPU cores. Each CPU-bound task blocks the event loop, so more workers compensate.
- **Memory constraints**: Each worker consumes its own copy of the application in memory. On memory-limited systems, reduce worker count accordingly. Monitor RSS per worker.
- **Container environments**: In Docker/Kubernetes, set worker count based on the container CPU limit, not the host CPU count. Detect available CPUs programmatically:

```python
import os

def get_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return (2 * cpu_count) + 1
```

In Kubernetes, read the CPU limit from the cgroup:

```python
import math
from pathlib import Path


def get_k8s_cpu_limit() -> int:
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0:
            return max(1, math.floor(quota / period))
    except (FileNotFoundError, ValueError):
        pass
    return os.cpu_count() or 1
```

Always validate worker count through load testing with realistic traffic patterns.

## Systemd Service File

Deploy Uvicorn (or Gunicorn + Uvicorn) as a systemd service for automatic startup, restart, and log management on Linux.

### Uvicorn Standalone Service

```ini
[Unit]
Description=Uvicorn ASGI Server for My API
After=network.target
Requires=network.target

[Service]
Type=simple
User=appuser
Group=appgroup
WorkingDirectory=/opt/my-api
Environment="PATH=/opt/my-api/.venv/bin"
Environment="APP_ENV=production"
Environment="APP_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/mydb"
ExecStart=/opt/my-api/.venv/bin/uvicorn app.main:app \
    --workers 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips "127.0.0.1" \
    --no-access-log \
    --log-level warning
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=my-api

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/my-api/data
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### Gunicorn + Uvicorn Service

```ini
[Unit]
Description=Gunicorn + Uvicorn ASGI Server for My API
After=network.target postgresql.service
Requires=network.target

[Service]
Type=notify
User=appuser
Group=appgroup
WorkingDirectory=/opt/my-api
Environment="PATH=/opt/my-api/.venv/bin"
Environment="APP_ENV=production"
ExecStart=/opt/my-api/.venv/bin/gunicorn app.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind unix:/run/my-api/gunicorn.sock \
    --timeout 120 \
    --graceful-timeout 30 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile -
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
Restart=always
RestartSec=5
RuntimeDirectory=my-api
StandardOutput=journal
StandardError=journal
SyslogIdentifier=my-api

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### Service Management Commands

```bash
# Enable and start
sudo systemctl enable my-api
sudo systemctl start my-api

# Check status and logs
sudo systemctl status my-api
sudo journalctl -u my-api -f

# Graceful reload (HUP signal)
sudo systemctl reload my-api

# Full restart
sudo systemctl restart my-api
```

## Nginx Reverse Proxy Configuration

### HTTP Proxy to Uvicorn TCP Socket

```nginx
upstream app_server {
    server 127.0.0.1:8000 fail_timeout=0;
}

server {
    listen 80;
    server_name api.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;

    client_max_body_size 10M;

    location / {
        proxy_pass http://app_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        proxy_buffering off;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://app_server;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400s;
    }

    # Static files (serve directly from Nginx)
    location /static/ {
        alias /opt/my-api/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

### Unix Socket Proxy

When Gunicorn binds to a Unix socket instead of a TCP port:

```nginx
upstream app_server {
    server unix:/run/my-api/gunicorn.sock fail_timeout=0;
}
```

Unix sockets avoid TCP overhead and are preferred when Nginx and the application run on the same host.

## Health Check Endpoint Patterns

Implement health and readiness endpoints for load balancers, orchestrators, and monitoring systems.

### Basic Health Check

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health_check():
    """Liveness probe. Returns 200 if the process is running."""
    return {"status": "healthy"}
```

### Readiness Check with Dependency Verification

```python
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/ready")
async def readiness_check():
    """Readiness probe. Verifies all dependencies are accessible."""
    checks = {}
    all_healthy = True

    # Check database
    try:
        async with app.state.db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        all_healthy = False

    # Check Redis
    try:
        await app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"
        all_healthy = False

    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        content={"status": "ready" if all_healthy else "not_ready", "checks": checks},
        status_code=status_code,
    )
```

### Kubernetes Probe Configuration

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3

startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 30
```

## Graceful Shutdown

### Signal Handling

Uvicorn handles `SIGTERM` and `SIGINT` for graceful shutdown by default. On receiving these signals, it:

1. Stops accepting new connections.
2. Waits for in-flight requests to complete (up to the graceful timeout).
3. Triggers ASGI lifespan shutdown events.
4. Exits the process.

With Gunicorn, configure the graceful timeout:

```bash
gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    --graceful-timeout 30 \
    --timeout 120
```

- `--graceful-timeout`: Seconds to wait for workers to finish after receiving `SIGHUP` or `SIGTERM`.
- `--timeout`: Maximum time a worker can be silent before being killed and restarted.

### Connection Draining in Application Code

Implement connection draining logic in the lifespan handler for long-lived connections:

```python
import asyncio
import signal
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    app.state.db_pool = await create_db_pool()
    app.state.shutting_down = False
    yield
    # Shutdown: drain connections
    app.state.shutting_down = True
    # Allow in-flight requests a grace period
    await asyncio.sleep(5)
    await app.state.db_pool.close()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def check_shutdown(request, call_next):
    if app.state.shutting_down:
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is shutting down"},
            headers={"Connection": "close", "Retry-After": "5"},
        )
    return await call_next(request)
```

### Kubernetes Graceful Shutdown

In Kubernetes, ensure the `terminationGracePeriodSeconds` exceeds the Gunicorn/Uvicorn graceful timeout:

```yaml
spec:
  terminationGracePeriodSeconds: 60
  containers:
    - name: api
      lifecycle:
        preStop:
          exec:
            command: ["sleep", "5"]  # Allow load balancer to deregister
```

## SSL/TLS Termination

### At Reverse Proxy Level (Recommended)

Terminate TLS at Nginx, Caddy, or a cloud load balancer. Run Uvicorn on plain HTTP internally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips "10.0.0.0/8"
```

This approach simplifies certificate management, enables centralized TLS configuration, and avoids duplicating certificate files to each application server.

### At Uvicorn Level

For direct TLS termination without a reverse proxy:

```bash
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 443 \
    --ssl-keyfile /etc/ssl/private/server.key \
    --ssl-certfile /etc/ssl/certs/server.crt \
    --ssl-ca-certs /etc/ssl/certs/ca-bundle.crt \
    --ssl-cert-reqs 0
```

SSL option reference:

| Option             | Description                                              |
|--------------------|----------------------------------------------------------|
| `--ssl-keyfile`    | Path to the SSL private key (PEM format).                |
| `--ssl-certfile`   | Path to the SSL certificate (PEM format).                |
| `--ssl-ca-certs`   | Path to CA bundle for client certificate verification.   |
| `--ssl-cert-reqs`  | Client cert requirement: `0` (none), `1` (optional), `2` (required). |

## Performance Tuning

### Uvicorn Performance Options

| Option                   | Default | Description                                                       |
|--------------------------|---------|-------------------------------------------------------------------|
| `--limit-concurrency`    | none    | Maximum number of concurrent connections. Reject excess with 503. |
| `--limit-max-requests`   | none    | Maximum requests per worker before recycling. Prevents memory leaks. |
| `--backlog`              | `2048`  | Maximum number of pending connections in the socket backlog.      |
| `--timeout-keep-alive`   | `5`     | Seconds to keep idle connections open for HTTP keep-alive.        |
| `--timeout-notify`       | `30`    | Seconds to notify workers before timeout.                         |

### Gunicorn Performance Options

```bash
gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    -w 4 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --keep-alive 5 \
    --backlog 2048 \
    --timeout 120
```

- `--max-requests`: Restart a worker after handling this many requests. Prevents slow memory leaks from accumulating.
- `--max-requests-jitter`: Add randomness to `--max-requests` to avoid all workers restarting simultaneously.
- `--keep-alive`: Seconds to wait for the next request on a keep-alive connection.
- `--backlog`: Maximum pending connections queue.

### Application-Level Optimizations

- Use connection pooling for databases (`asyncpg` pool, SQLAlchemy async pool).
- Enable HTTP response compression via middleware (`GZipMiddleware`).
- Cache frequently accessed data in Redis or in-process caches.
- Use streaming responses (`StreamingResponse`) for large payloads.
- Profile with `py-spy` or `yappi` to identify event-loop-blocking code.

## Monitoring and Metrics

### Prometheus Metrics Integration

Install the Prometheus client and expose metrics from the FastAPI application:

```bash
uv add prometheus-client prometheus-fastapi-instrumentator
```

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Initialize and expose /metrics endpoint
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    excluded_handlers=["/health", "/ready", "/metrics"],
    env_var_name="ENABLE_METRICS",
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")
```

### Key Metrics to Monitor

- **Request rate**: Requests per second by endpoint and status code.
- **Request latency**: p50, p95, p99 response times.
- **Error rate**: 4xx and 5xx responses as a percentage of total requests.
- **Active connections**: Number of concurrent connections per worker.
- **Worker memory**: RSS memory per worker process (detect memory leaks).
- **Event loop lag**: Time the event loop is blocked (indicates CPU-bound work).

### Structured Logging for Observability

Produce JSON-structured logs for aggregation in ELK, Datadog, or similar platforms:

```python
import logging
import json
import sys

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "app.logging.JSONFormatter",
        },
    },
    "handlers": {
        "default": {
            "formatter": "json",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}
```

## Zero-Downtime Deployment Strategies

### Rolling Restart with Gunicorn

Send `SIGHUP` to the Gunicorn master process to gracefully restart all workers one at a time:

```bash
kill -HUP $(cat /run/my-api/gunicorn.pid)
# or
sudo systemctl reload my-api
```

Gunicorn spawns new workers before killing old ones, maintaining availability throughout the restart.

### Blue-Green Deployment

1. Deploy the new version to a separate set of instances (green).
2. Run health checks against the green instances.
3. Switch the load balancer to point to green.
4. Drain and shut down the old instances (blue).

### Rolling Update in Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-api
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      terminationGracePeriodSeconds: 60
      containers:
        - name: api
          image: my-api:v2.0.0
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests:
              cpu: "500m"
              memory: "256Mi"
            limits:
              cpu: "1000m"
              memory: "512Mi"
          lifecycle:
            preStop:
              exec:
                command: ["sleep", "5"]
```

Kubernetes waits for the readiness probe to pass on new pods before terminating old ones, ensuring zero-downtime transitions.

## Environment Variable Configuration for Production

Centralize all runtime configuration in environment variables, loaded via `pydantic-settings`:

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=1, ge=1)
    log_level: str = "info"
    debug: bool = False

    # Security
    allowed_origins: list[str] = ["https://example.com"]
    proxy_headers: bool = True
    forwarded_allow_ips: str = "127.0.0.1"

    # Database
    database_url: str
    db_pool_min: int = 5
    db_pool_max: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Secrets
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
```

Example `.env` for production:

```bash
APP_HOST=0.0.0.0
APP_PORT=8000
APP_WORKERS=4
APP_LOG_LEVEL=warning
APP_DEBUG=false
APP_ALLOWED_ORIGINS=["https://example.com","https://app.example.com"]
APP_PROXY_HEADERS=true
APP_FORWARDED_ALLOW_IPS=10.0.0.0/8
APP_DATABASE_URL=postgresql+asyncpg://user:password@db.internal:5432/mydb
APP_DB_POOL_MIN=10
APP_DB_POOL_MAX=50
APP_REDIS_URL=redis://redis.internal:6379/0
APP_SECRET_KEY=<generated-secret>
APP_JWT_ALGORITHM=HS256
APP_JWT_EXPIRE_MINUTES=15
```

Launch using the settings object:

```python
import uvicorn
from app.config import ServerSettings

settings = ServerSettings()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        proxy_headers=settings.proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
    )
```

## Production Deployment Checklist

- [ ] Set `--workers` based on CPU cores: `(2 * cores) + 1`
- [ ] Enable `--proxy-headers` and configure `--forwarded-allow-ips` when behind a reverse proxy
- [ ] Disable `--access-log` if the reverse proxy already logs requests
- [ ] Set `--log-level` to `warning` or `error` for production
- [ ] Configure `--limit-max-requests` to recycle workers periodically
- [ ] Set `--timeout-keep-alive` to match the reverse proxy keep-alive timeout
- [ ] Implement `/health` and `/ready` endpoints
- [ ] Use lifespan handlers for resource initialization and cleanup
- [ ] Terminate TLS at the reverse proxy level
- [ ] Run the application as a non-root user
- [ ] Set `--backlog` appropriately for expected connection rates
- [ ] Enable Prometheus metrics for monitoring
- [ ] Configure structured JSON logging for log aggregation
- [ ] Test graceful shutdown behavior under load
- [ ] Validate zero-downtime deployment process
