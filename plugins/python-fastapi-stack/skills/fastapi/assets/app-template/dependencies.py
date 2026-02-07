"""Shared FastAPI dependencies.

Import and use these dependencies across routers via `Depends()` or
`Annotated` type aliases.
"""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (app-scoped, created once)."""
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------

async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session scoped to the current request.

    The session is committed on success and rolled back on exception.
    Requires `app.state.async_session` to be initialized in the lifespan.
    """
    async_session = request.app.state.async_session
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbSession = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class PaginationParams:
    """Common pagination query parameters."""

    def __init__(
        self,
        offset: int = Query(0, ge=0, description="Number of items to skip"),
        limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    ):
        self.offset = offset
        self.limit = limit


Pagination = Annotated[PaginationParams, Depends()]


# ---------------------------------------------------------------------------
# Common query parameters
# ---------------------------------------------------------------------------

class CommonQueryParams:
    """Reusable query parameters shared across list endpoints."""

    def __init__(
        self,
        q: str | None = Query(None, max_length=200, description="Search query"),
        sort_by: str = Query("created_at", description="Field to sort by"),
        order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    ):
        self.q = q
        self.sort_by = sort_by
        self.order = order


CommonQuery = Annotated[CommonQueryParams, Depends()]
