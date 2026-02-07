---
name: dependency-injection
description: >
  FastAPI dependency injection reference covering dependency chains, database session
  management, authentication and permission dependencies, scoping, parameterized and
  class-based dependencies, testing overrides, global dependencies, and caching.
version: 1.0.0
---

# Dependency Injection Reference

## How FastAPI DI Works

FastAPI resolves dependencies declared with `Depends()` at request time. Each dependency
is a callable (function, async function, or class with `__call__`) whose parameters are
themselves resolved recursively. The framework builds a dependency graph per request and
caches results within that request so each dependency runs at most once.

```python
from fastapi import Depends
from typing import Annotated

async def dep_a() -> int:
    return 42

async def dep_b(a: Annotated[int, Depends(dep_a)]) -> str:
    return f"value-{a}"

@app.get("/example")
async def example(b: Annotated[str, Depends(dep_b)]):
    # dep_a runs once, dep_b receives its result
    return {"result": b}
```

## Dependency Chains and Composition

Build complex dependency trees by composing simple, single-responsibility dependencies:

```python
from app.config import Settings

def get_settings() -> Settings:
    return Settings()

async def get_db(settings: Annotated[Settings, Depends(get_settings)]):
    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        yield session

async def get_current_user(
    db: Annotated[AsyncSession, Depends(get_db)],
    token: Annotated[str, Depends(oauth2_scheme)],
) -> User:
    payload = decode_token(token)
    user = await db.get(User, payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_current_active_user(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    return user
```

Create type aliases for commonly used dependencies to keep route signatures concise:

```python
DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
ActiveUser = Annotated[User, Depends(get_current_active_user)]

@app.get("/me")
async def read_me(user: ActiveUser):
    return user
```

## Database Session with Yield

The yield dependency pattern is the standard approach for managing database sessions.
Code before `yield` runs during request processing; code after `yield` runs during
cleanup (after the response is sent).

### Async Session

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### Sync Session (for sync database drivers)

```python
from sqlalchemy.orm import Session, sessionmaker

SessionLocal = sessionmaker(bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
```

### Transaction Management Patterns

Explicit transaction boundaries for operations requiring atomicity:

```python
async def get_db_with_transaction() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        async with session.begin():
            yield session
            # commit happens automatically at the end of the `begin()` block
            # rollback happens automatically on exception
```

## Authentication Dependencies

### OAuth2 Password Bearer

```python
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: DbSession,
) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Could not validate token")

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
```

### API Key Authentication

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(
    api_key: Annotated[str, Depends(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    if api_key not in settings.valid_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

### Optional Authentication

Allow both authenticated and unauthenticated access:

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

async def get_optional_user(
    token: Annotated[str | None, Depends(oauth2_scheme_optional)],
    db: DbSession,
) -> User | None:
    if token is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return await db.get(User, payload["sub"])
    except jwt.InvalidTokenError:
        return None
```

## Permission Dependencies

### Role-Based Access Control

```python
from enum import Enum

class Role(str, Enum):
    user = "user"
    admin = "admin"
    superadmin = "superadmin"

def require_role(*roles: Role):
    async def dependency(user: CurrentUser):
        if user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role {user.role} not authorized. Required: {', '.join(roles)}",
            )
        return user
    return dependency

# Usage
AdminUser = Annotated[User, Depends(require_role(Role.admin, Role.superadmin))]

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, admin: AdminUser, db: DbSession):
    ...
```

### Resource-Level Permissions

```python
async def get_owned_item(
    item_id: int,
    user: CurrentUser,
    db: DbSession,
) -> Item:
    item = await db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if item.owner_id != user.id and user.role != Role.admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    return item

OwnedItem = Annotated[Item, Depends(get_owned_item)]

@app.put("/items/{item_id}")
async def update_item(item: OwnedItem, data: ItemUpdate, db: DbSession):
    ...
```

## Request-Scoped vs. App-Scoped Dependencies

By default, every dependency is **request-scoped**: it runs once per request and the
result is cached for that request's dependency graph. If two route parameters depend on
the same function, that function executes only once.

**App-scoped** resources (connection pools, HTTP clients, caches) should be initialized
in the lifespan and stored on `app.state`, then accessed via a dependency:

```python
from fastapi import Request

def get_redis(request: Request) -> Redis:
    return request.app.state.redis

def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client
```

This avoids re-creating expensive resources on each request.

## Parameterized Dependencies (Factory Pattern)

Create configurable dependencies using closures or functools.partial:

```python
def require_header(header_name: str, expected_value: str):
    async def dependency(request: Request):
        value = request.headers.get(header_name)
        if value != expected_value:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {header_name} header",
            )
        return value
    return dependency

# Usage
@app.get("/internal", dependencies=[Depends(require_header("X-Internal", "true"))])
async def internal_endpoint():
    ...
```

### Pagination Factory

```python
def paginator(max_limit: int = 100, default_limit: int = 20):
    def dependency(
        offset: int = Query(0, ge=0),
        limit: int = Query(default_limit, ge=1, le=max_limit),
    ) -> dict[str, int]:
        return {"offset": offset, "limit": limit}
    return dependency

default_pagination = paginator()
large_pagination = paginator(max_limit=500, default_limit=50)

@app.get("/items")
async def list_items(pagination: Annotated[dict, Depends(default_pagination)]):
    ...

@app.get("/logs")
async def list_logs(pagination: Annotated[dict, Depends(large_pagination)]):
    ...
```

## Class-Based Dependencies

Use classes with `__call__` for dependencies that carry configuration state:

```python
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.store: dict[str, list[float]] = {}

    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window

        self.store.setdefault(client_ip, [])
        self.store[client_ip] = [t for t in self.store[client_ip] if t > window_start]

        if len(self.store[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        self.store[client_ip].append(now)

# Create instances with different configurations
strict_limiter = RateLimiter(max_requests=10, window_seconds=60)
relaxed_limiter = RateLimiter(max_requests=100, window_seconds=60)

@app.post("/login", dependencies=[Depends(strict_limiter)])
async def login():
    ...

@app.get("/items", dependencies=[Depends(relaxed_limiter)])
async def list_items():
    ...
```

### Class Dependencies with Init Parameters from DI

```python
class ItemService:
    def __init__(self, db: DbSession, user: CurrentUser):
        self.db = db
        self.user = user

    async def list_items(self) -> list[Item]:
        result = await self.db.execute(
            select(Item).where(Item.owner_id == self.user.id)
        )
        return result.scalars().all()

# FastAPI resolves db and user automatically
@app.get("/items")
async def list_items(service: Annotated[ItemService, Depends()]):
    return await service.list_items()
```

When `Depends()` is called with no arguments on a class, FastAPI inspects the `__init__`
signature and resolves each parameter as a dependency.

## Testing with dependency_overrides

Override dependencies in tests to inject mocks, test databases, or fixed values:

```python
from fastapi.testclient import TestClient

def test_list_items():
    # Override the database dependency
    async def override_get_db():
        async with test_session_factory() as session:
            yield session

    # Override authentication
    async def override_get_current_user():
        return User(id=1, email="test@example.com", role=Role.user)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    client = TestClient(app)
    response = client.get("/items")
    assert response.status_code == 200

    # Always clean up
    app.dependency_overrides.clear()
```

### Fixture-Based Override Pattern

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    async def mock_db():
        async with test_session() as session:
            yield session

    app.dependency_overrides[get_db] = mock_db
    app.dependency_overrides[get_current_user] = lambda: mock_user

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
```

### Async Test Override

```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.fixture
async def async_client():
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.clear()

@pytest.mark.anyio
async def test_create_item(async_client):
    response = await async_client.post("/items", json={"name": "Test", "price": 9.99})
    assert response.status_code == 201
```

## Global Dependencies

Apply dependencies to every route on an application or router without listing them on
each endpoint:

### Application-Level

```python
app = FastAPI(dependencies=[Depends(verify_api_key)])
```

### Router-Level

```python
router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(require_role(Role.admin))],
)
```

Global dependencies run for every matched route. They cannot return values to the
route function (since they are not declared as parameters), but they can raise exceptions
to block access or add data to `request.state`.

### Combining Global and Per-Route Dependencies

```python
# All routes require a valid API key (global)
app = FastAPI(dependencies=[Depends(verify_api_key)])

# Admin routes additionally require admin role (router-level)
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(require_role(Role.admin))],
)

# A specific route adds yet another dependency
@admin_router.delete("/users/{user_id}", dependencies=[Depends(audit_log)])
async def delete_user(user_id: int, db: DbSession):
    ...
```

## Lazy and Cached Dependencies

### Cached Dependencies (Default Behavior)

FastAPI caches dependency results per request. If `get_db` appears in multiple places in
the dependency graph, it executes once and the same session is shared:

```python
async def get_user_service(db: DbSession) -> UserService:
    return UserService(db)

async def get_item_service(db: DbSession) -> ItemService:
    return ItemService(db)

@app.get("/dashboard")
async def dashboard(
    user_svc: Annotated[UserService, Depends(get_user_service)],
    item_svc: Annotated[ItemService, Depends(get_item_service)],
):
    # Both services share the same db session
    ...
```

### Disabling Cache

Force a dependency to run every time by passing `use_cache=False`:

```python
@app.get("/example")
async def example(
    ts1: Annotated[float, Depends(get_timestamp)],
    ts2: Annotated[float, Depends(get_timestamp, use_cache=False)],
):
    # ts1 and ts2 may differ because ts2 is not cached
    ...
```

### Lazy Initialization Pattern

Defer expensive initialization until first use within a request:

```python
class LazyResource:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()

lazy = LazyResource()

async def get_external_client() -> httpx.AsyncClient:
    return await lazy.get_client()
```

For truly app-scoped lazy resources, initialize in the lifespan and store on
`app.state`. The dependency pattern above is better suited for request-scoped resources
that may not be needed on every request.
