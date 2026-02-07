---
name: database
description: >
  Guides the agent through async database integration with SQLAlchemy and Alembic
  migrations for FastAPI applications. Triggered when users ask to "set up a database",
  "create database models", "add SQLAlchemy", "create migrations", "run Alembic",
  "connect to PostgreSQL", "add a database layer", "create CRUD operations",
  "set up async database", or mention SQLAlchemy, Alembic, ORM, database models,
  async database, connection pool, or database migrations.
version: 1.0.0
---

# Database Integration (SQLAlchemy Async + Alembic)

## Overview

Use **SQLAlchemy 2.0+** with async support as the database toolkit for FastAPI applications. SQLAlchemy provides both ORM (Object-Relational Mapping) and Core (SQL expression) layers. Pair with **Alembic** for schema migrations.

Key packages:

```bash
uv add "sqlalchemy[asyncio]" alembic asyncpg  # PostgreSQL
# or
uv add "sqlalchemy[asyncio]" alembic aiosqlite  # SQLite
```

- `sqlalchemy[asyncio]` -- async engine and session support
- `asyncpg` -- high-performance async PostgreSQL driver
- `aiosqlite` -- async SQLite driver
- `alembic` -- database migration tool

## Async Engine and Session

### Engine Setup

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/mydb"
# For SQLite: "sqlite+aiosqlite:///./app.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,           # Set True for SQL query logging
    pool_size=5,          # Number of persistent connections
    max_overflow=10,      # Additional connections allowed beyond pool_size
    pool_pre_ping=True,   # Verify connections before use
    pool_recycle=3600,    # Recycle connections after 1 hour
)

async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,  # Prevent lazy-load after commit in async
)
```

### FastAPI Integration with Lifespan

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Engine is created at module level; dispose on shutdown
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

Use `Depends(get_db)` in route functions to inject a session per request.

## Model Definition

### Declarative Base

```python
from datetime import datetime

from sqlalchemy import String, Text, ForeignKey, func
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
    )
```

### Model Examples

```python
class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)

    # Relationships
    posts: Mapped[list["Post"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",
    )


class Post(TimestampMixin, Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    author: Mapped["User"] = relationship(back_populates="posts")
```

Use `Mapped[type]` with `mapped_column()` for all column definitions (SQLAlchemy 2.0 style). Avoid the legacy `Column()` syntax.

## CRUD Operations

### Repository Pattern

```python
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: int) -> User | None:
        return await self.session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_users(
        self, *, offset: int = 0, limit: int = 20
    ) -> list[User]:
        stmt = select(User).offset(offset).limit(limit).order_by(User.id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, **kwargs) -> User:
        user = User(**kwargs)
        self.session.add(user)
        await self.session.flush()  # Assign ID without committing
        return user

    async def update(self, user: User, **kwargs) -> User:
        for key, value in kwargs.items():
            setattr(user, key, value)
        await self.session.flush()
        return user

    async def delete(self, user: User) -> None:
        await self.session.delete(user)

    async def count(self) -> int:
        stmt = select(func.count()).select_from(User)
        result = await self.session.execute(stmt)
        return result.scalar_one()
```

### Using in FastAPI Routes

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/", status_code=201)
async def create_user(
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    repo = UserRepository(db)
    existing = await repo.get_by_email(data.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = await repo.create(**data.model_dump())
    return user
```

## Query Patterns

### Eager Loading (Avoid N+1)

```python
from sqlalchemy.orm import selectinload, joinedload

# selectinload: separate IN query (best for collections)
stmt = select(User).options(selectinload(User.posts)).where(User.id == user_id)

# joinedload: LEFT JOIN (best for single relationships)
stmt = select(Post).options(joinedload(Post.author)).where(Post.id == post_id)
```

Always use eager loading when accessing relationships in async contexts. Lazy loading raises errors under async sessions.

### Filtering and Ordering

```python
from sqlalchemy import and_, or_, desc

# Complex filters
stmt = (
    select(User)
    .where(
        and_(
            User.is_active == True,
            or_(
                User.name.ilike(f"%{query}%"),
                User.email.ilike(f"%{query}%"),
            ),
        )
    )
    .order_by(desc(User.created_at))
    .offset(offset)
    .limit(limit)
)
```

### Pagination

```python
from pydantic import BaseModel


class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int
    offset: int
    limit: int

    @property
    def has_more(self) -> bool:
        return self.offset + self.limit < self.total


async def paginate(
    session: AsyncSession,
    stmt,
    *,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list, int]:
    # Count query
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar_one()

    # Data query
    result = await session.execute(stmt.offset(offset).limit(limit))
    items = list(result.scalars().all())

    return items, total
```

## Alembic Migrations

### Setup

```bash
# Initialize Alembic
uv run alembic init alembic

# For async support, use the async template
uv run alembic init -t async alembic
```

Configure `alembic/env.py`:

```python
# alembic/env.py
from app.database import Base, DATABASE_URL
from app.models import User, Post  # Import all models

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("+asyncpg", ""))

target_metadata = Base.metadata
```

For the async template, update `run_async_migrations()` in `env.py` to use your async engine.

### Migration Commands

```bash
# Generate a migration from model changes
uv run alembic revision --autogenerate -m "add users table"

# Apply all pending migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1

# Show current migration status
uv run alembic current

# Show migration history
uv run alembic history
```

### Migration Best Practices

1. **Always review auto-generated migrations** before applying -- autogenerate cannot detect all changes (renamed columns, data migrations).
2. **Test migrations both ways** -- run `upgrade` and `downgrade` to verify reversibility.
3. **Use descriptive revision messages** -- `add_users_table` not `update`.
4. **Never edit applied migrations** -- create new migrations instead.
5. **Include migrations in version control** -- the `alembic/versions/` directory should be committed.

## Connection Pool Tuning

| Parameter       | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `pool_size`     | 5       | Number of persistent connections in the pool      |
| `max_overflow`  | 10      | Extra connections allowed beyond `pool_size`       |
| `pool_timeout`  | 30      | Seconds to wait for a connection before error      |
| `pool_recycle`  | -1      | Seconds before a connection is recycled (set for PG) |
| `pool_pre_ping` | False   | Test connections before checkout (set True for prod) |

Production recommendation for a 4-worker FastAPI app:

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,          # 5 per worker = 20 total connections
    max_overflow=10,       # Burst to 15 per worker
    pool_pre_ping=True,    # Handle dropped connections
    pool_recycle=3600,     # Recycle hourly
)
```

Total max connections = `workers * (pool_size + max_overflow)`. Ensure the database `max_connections` setting accommodates this.

## Testing with Database

### In-Memory SQLite for Unit Tests

```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.database import Base, get_db
from app.main import create_app


@pytest.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def app(db_session):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    yield app
    app.dependency_overrides.clear()
```

### Transaction Rollback Pattern

Wrap each test in a transaction that always rolls back for isolation:

```python
@pytest.fixture
async def db_session(db_engine):
    async with db_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        yield session
        await trans.rollback()
```

## Cross-References

- For Pydantic request/response models, consult the `pydantic` skill.
- For FastAPI routing and dependency injection, consult the `app-scaffolding` skill.
- For async patterns and error handling, consult the `async-patterns` skill.
- For Docker Compose with PostgreSQL, consult the `docker-build` skill.
- For test fixtures and pytest patterns, consult the `test-runner` skill.
