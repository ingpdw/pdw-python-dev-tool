---
name: middleware
description: >
  FastAPI middleware reference covering CORSMiddleware configuration, custom middleware
  with BaseHTTPMiddleware and pure ASGI, request timing, structured logging, rate
  limiting, authentication, middleware ordering, GZip, and trusted hosts.
version: 1.0.0
---

# Middleware Reference

## Middleware Execution Flow

Middleware wraps the application in layers. Each request passes through middleware in
the order they are **added** (top to bottom), and each response passes back through
them in **reverse order**:

```
Request  --> Middleware A --> Middleware B --> Route Handler
Response <-- Middleware A <-- Middleware B <--
```

When using `app.add_middleware()`, FastAPI prepends each middleware to the stack, so the
**last added** middleware is the **outermost** layer (first to process requests). Plan
the registration order accordingly:

```python
# Execution order for requests: CORS -> Logging -> Timing -> Route
# Registration order (reversed):
app.add_middleware(TimingMiddleware)      # innermost
app.add_middleware(LoggingMiddleware)     # middle
app.add_middleware(CORSMiddleware, ...)   # outermost
```

## CORSMiddleware Deep Dive

Cross-Origin Resource Sharing (CORS) controls which origins can access the API from
a browser.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com", "https://staging.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Total-Count"],
    max_age=600,  # cache preflight for 10 minutes
)
```

### Configuration Parameters

| Parameter           | Description                                          | Default |
|---------------------|------------------------------------------------------|---------|
| `allow_origins`     | List of allowed origin URLs                          | `[]`    |
| `allow_origin_regex`| Regex pattern for allowed origins                    | `None`  |
| `allow_methods`     | HTTP methods to allow                                | `["GET"]` |
| `allow_headers`     | Request headers to allow                             | `[]`    |
| `allow_credentials` | Allow cookies and Authorization headers              | `False` |
| `expose_headers`    | Response headers accessible to the browser           | `[]`    |
| `max_age`           | Seconds to cache preflight response                  | `600`   |

### Common Pitfalls

- **Wildcard with credentials**: `allow_origins=["*"]` with `allow_credentials=True`
  is rejected by browsers. List explicit origins instead.
- **Missing preflight methods**: Ensure `OPTIONS` is included in `allow_methods` (it is
  added automatically by `CORSMiddleware`).
- **Origin vs. URL**: Origins include scheme and host only (`https://example.com`), not
  paths.

### Dynamic Origin Validation

Use `allow_origin_regex` for patterns:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.example\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Custom Middleware with BaseHTTPMiddleware

`BaseHTTPMiddleware` is the simplest way to write custom middleware. It provides a
`dispatch` method that receives the request and a `call_next` callable:

```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class CustomHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Custom-Header"] = "my-value"
        return response

app.add_middleware(CustomHeaderMiddleware)
```

### Limitations of BaseHTTPMiddleware

- Reads the entire response body into memory before returning, which prevents true
  streaming.
- Creates a new `anyio` task for each request, adding overhead.
- Cannot catch exceptions raised inside `call_next` in all cases.

For high-throughput or streaming scenarios, prefer pure ASGI middleware.

## Pure ASGI Middleware

Write middleware as an ASGI application for maximum performance and full streaming
support:

```python
from starlette.types import ASGIApp, Receive, Scope, Send

class PureASGIMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Pre-processing
        scope["state"]["request_id"] = generate_request_id()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-request-id"] = scope["state"]["request_id"].encode()
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)

app.add_middleware(PureASGIMiddleware)
```

### When to Use Pure ASGI

- Streaming responses (SSE, file downloads) where body buffering is unacceptable
- High-throughput APIs where per-request overhead matters
- Middleware that needs to intercept both HTTP and WebSocket connections
- Middleware that must modify raw ASGI messages

## Request Timing Middleware

Track and expose request processing time:

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response

app.add_middleware(TimingMiddleware)
```

### Pure ASGI Timing (Streaming-Safe)

```python
import time
from starlette.types import ASGIApp, Receive, Scope, Send

class ASGITimingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()

        async def send_with_timing(message):
            if message["type"] == "http.response.start":
                duration_ms = (time.perf_counter() - start) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time-ms", f"{duration_ms:.1f}".encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_timing)
```

## Logging Middleware

Structured logging for every request/response cycle:

```python
import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.access")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 1),
                "client_ip": request.client.host if request.client else None,
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response
```

Pair this with a structured logging formatter (such as `python-json-logger` or
`structlog`) to emit JSON logs suitable for aggregation in production.

## Rate Limiting Patterns

### In-Memory Token Bucket

Suitable for single-process development and testing:

```python
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.clients: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60

        # Remove expired entries
        self.clients[client_ip] = [
            t for t in self.clients[client_ip] if t > window_start
        ]

        if len(self.clients[client_ip]) >= self.rpm:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
                headers={"Retry-After": "60"},
            )

        self.clients[client_ip].append(now)
        return await call_next(request)
```

### Redis-Backed Rate Limiting

For multi-process / multi-node deployments, use Redis:

```python
import redis.asyncio as redis

class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str, rpm: int = 60):
        super().__init__(app)
        self.redis = redis.from_url(redis_url)
        self.rpm = rpm

    async def dispatch(self, request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{client_ip}"

        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 60)

        if count > self.rpm:
            return JSONResponse(status_code=429, content={"detail": "Too many requests"})

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.rpm - count))
        return response
```

## Authentication Middleware

### JWT Verification Middleware

```python
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str, algorithm: str = "HS256"):
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def dispatch(self, request, call_next):
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing token"})

        token = auth_header.removeprefix("Bearer ")
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            request.state.user = payload
        except jwt.InvalidTokenError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})

        return await call_next(request)
```

### API Key Middleware

```python
class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_keys: set[str]):
        super().__init__(app)
        self.api_keys = api_keys

    async def dispatch(self, request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key not in self.api_keys:
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

        return await call_next(request)
```

Prefer dependency-based authentication over middleware when fine-grained per-route
control is needed.

## GZip Middleware

Compress responses larger than a minimum size threshold:

```python
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=500)  # bytes
```

GZip middleware should be one of the outermost layers so it compresses the final
response body. Do not apply it if a reverse proxy (nginx, Cloudflare) already handles
compression.

## Trusted Host Middleware

Prevent host header attacks by validating the `Host` header:

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.example.com", "*.example.com"],
)
```

In development, add `"localhost"` and `"127.0.0.1"` to the allowed hosts list, or
disable the middleware entirely.

## Middleware Registration Patterns

### Conditional Middleware Based on Environment

```python
from app.config import get_settings

settings = get_settings()

if settings.debug:
    app.add_middleware(TimingMiddleware)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if not settings.debug:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(GZipMiddleware, minimum_size=500)
```

### Recommended Production Middleware Stack

Register in this order (last registered = outermost = first to execute):

```python
# 1. Timing (innermost -- measures actual handler time)
app.add_middleware(TimingMiddleware)

# 2. Rate limiting
app.add_middleware(RateLimitMiddleware, requests_per_minute=120)

# 3. Logging
app.add_middleware(LoggingMiddleware)

# 4. GZip compression
app.add_middleware(GZipMiddleware, minimum_size=500)

# 5. CORS (outermost -- must handle preflight before other middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Trusted hosts (outermost -- reject bad hosts immediately)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts,
)
```
