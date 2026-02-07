---
name: fastapi
description: >
  Guides the agent through building FastAPI applications, including API routes,
  request/response models, path and query parameters, dependency injection,
  middleware, error handling, and project structure. Triggered when the user asks
  to "create a FastAPI app", "add an API endpoint", "create a router",
  "add middleware", "implement dependency injection", "handle errors",
  "set up CORS", "create background tasks", "implement WebSocket",
  "structure a FastAPI project", or "add authentication".
version: 1.0.0
---

# FastAPI Skill

## Overview

FastAPI is a modern, high-performance async Python web framework for building APIs. It
leverages Python type hints and Pydantic for automatic validation, serialization, and
OpenAPI documentation generation. FastAPI runs on ASGI servers (Uvicorn, Hypercorn) and
provides first-class support for async/await, dependency injection, and WebSockets.

Key characteristics:

- Automatic interactive API docs (Swagger UI at `/docs`, ReDoc at `/redoc`)
- Type-driven request validation and response serialization via Pydantic
- Native async support with full sync fallback
- Dependency injection system for shared logic and resource management
- Standards-based: OpenAPI 3.1, JSON Schema

## Project Structure

Organize FastAPI projects with a clear separation of concerns:

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application factory and lifespan
│   ├── config.py            # Settings via pydantic-settings
│   ├── dependencies.py      # Shared dependencies
│   ├── models.py            # SQLAlchemy / ORM models
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── routers/
│   │   ├── __init__.py      # Router registration
│   │   ├── users.py
│   │   ├── items.py
│   │   └── health.py
│   ├── services/            # Business logic layer
│   ├── middleware/           # Custom middleware
│   └── exceptions.py        # Custom exception handlers
├── tests/
│   ├── conftest.py
│   ├── test_users.py
│   └── test_items.py
├── alembic/                 # Database migrations
├── pyproject.toml
└── .env
```

See `assets/app-template/` for a ready-to-use scaffold.

## Application Factory and Lifespan

Use the async context manager lifespan pattern (the `on_event` decorator is deprecated):

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    app.state.db_pool = await create_db_pool()
    app.state.http_client = httpx.AsyncClient()
    yield
    # Shutdown: release resources
    await app.state.http_client.aclose()
    await app.state.db_pool.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="My API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(users_router)
    app.include_router(items_router)
    return app

app = create_app()
```

## Route Definition

Define routes with HTTP method decorators. Use type-annotated parameters for automatic
validation and documentation.

### Path Parameters

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int) -> User:
    ...
```

### Query Parameters

```python
@app.get("/items")
async def list_items(skip: int = 0, limit: int = 20, q: str | None = None):
    ...
```

### Request Body

```python
from pydantic import BaseModel

class ItemCreate(BaseModel):
    name: str
    price: float
    description: str | None = None

@app.post("/items", status_code=201)
async def create_item(item: ItemCreate) -> Item:
    ...
```

### Multiple Parameter Sources

```python
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,                          # path
    item: ItemUpdate,                      # body
    q: str | None = None,                  # query
    x_token: Annotated[str, Header()],     # header
):
    ...
```

## Response Models

Control response shape and status codes:

```python
@app.post("/users", response_model=UserOut, status_code=201)
async def create_user(user: UserCreate):
    ...

@app.get("/report", response_class=HTMLResponse)
async def get_report():
    return "<html>...</html>"
```

Use `response_model_exclude_unset=True` to omit fields not explicitly set, and
`response_model_exclude={"password"}` to strip sensitive fields.

## Router Organization

Split endpoints into routers for modularity:

```python
# app/routers/users.py
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
async def list_users():
    ...

@router.get("/{user_id}")
async def get_user(user_id: int):
    ...
```

Register routers in the application factory:

```python
from app.routers import users, items

app.include_router(users.router)
app.include_router(items.router, prefix="/api/v1")
```

Apply shared dependencies to all routes on a router:

```python
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)
```

## Dependency Injection

Use `Depends()` to declare reusable, composable dependencies. Dependencies can be
functions, async functions, or callable classes.

```python
from fastapi import Depends
from typing import Annotated

async def get_db():
    async with async_session() as session:
        yield session

DbSession = Annotated[AsyncSession, Depends(get_db)]

async def get_current_user(db: DbSession, token: str = Depends(oauth2_scheme)) -> User:
    ...

CurrentUser = Annotated[User, Depends(get_current_user)]

@app.get("/me")
async def read_me(user: CurrentUser):
    return user
```

Yield dependencies execute cleanup after the response is sent, making them ideal for
database sessions, file handles, and other resources that require teardown.

See `references/dependency-injection.md` for advanced patterns including parameterized
dependencies, class-based dependencies, and testing overrides.

## Request and Response Models with Pydantic

Define strict, validated schemas using Pydantic models. Separate input and output
schemas to control what is accepted vs. returned:

```python
class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

Cross-reference: see the **pydantic** skill for comprehensive model patterns, custom
validators, and serialization options.

## Lifespan Events

The lifespan async context manager replaces the deprecated `@app.on_event("startup")`
and `@app.on_event("shutdown")` decorators. Everything before `yield` runs at startup;
everything after runs at shutdown.

Store shared resources on `app.state` so dependencies can access them:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await aioredis.from_url("redis://localhost")
    yield
    await app.state.redis.close()
```

## Background Tasks

Enqueue lightweight work to run after the response is returned:

```python
from fastapi import BackgroundTasks

def send_notification(email: str, message: str):
    # blocking or async work
    ...

@app.post("/orders")
async def create_order(order: OrderCreate, background_tasks: BackgroundTasks):
    result = await process_order(order)
    background_tasks.add_task(send_notification, order.email, "Order confirmed")
    return result
```

For heavier workloads, offload to a task queue (Celery, ARQ, or Dramatiq) instead.

## Error Handling

### HTTPException

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    item = await fetch_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

### Custom Exception Handlers

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

class ItemNotFoundError(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Item {exc.item_id} not found"},
    )

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )
```

## CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Never use `allow_origins=["*"]` with `allow_credentials=True` in production. See
`references/middleware.md` for detailed CORS guidance and other middleware patterns.

## Testing

Use `TestClient` for synchronous tests or `httpx.AsyncClient` for async tests.
Override dependencies to inject mocks:

```python
from fastapi.testclient import TestClient

def test_read_items():
    app.dependency_overrides[get_db] = lambda: mock_db
    client = TestClient(app)
    response = client.get("/items")
    assert response.status_code == 200
    app.dependency_overrides.clear()
```

Async test with `httpx`:

```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.anyio
async def test_create_item():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/items", json={"name": "Widget", "price": 9.99})
    assert response.status_code == 201
```

## Further Reading

- `references/routing-patterns.md` -- advanced routing, file uploads, WebSockets, SSE
- `references/middleware.md` -- middleware patterns, ordering, custom middleware
- `references/dependency-injection.md` -- DI composition, testing, scoping
- `assets/app-template/` -- production-ready project scaffold

Cross-references:
- **pydantic** skill -- model definitions, validators, serialization
- **async-patterns** skill -- async/await best practices, concurrency
- **uvicorn** skill -- ASGI server configuration and deployment
