"""Production-ready FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import Settings
from app.routers import health, items, users


def get_settings() -> Settings:
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown resources.

    Initialize shared resources (database pools, HTTP clients, caches) before
    yield and clean them up after yield.
    """
    settings = get_settings()

    # -- Startup --
    # Example: initialize a database connection pool
    # engine = create_async_engine(settings.database_url)
    # app.state.db_engine = engine
    # app.state.async_session = async_sessionmaker(engine, expire_on_commit=False)

    # Example: initialize an HTTP client for external API calls
    # app.state.http_client = httpx.AsyncClient(timeout=30)

    yield

    # -- Shutdown --
    # Example: close the HTTP client
    # await app.state.http_client.aclose()

    # Example: dispose of the database engine
    # await app.state.db_engine.dispose()


def create_app() -> FastAPI:
    """Application factory.

    Create and configure the FastAPI application with routers, middleware, and
    exception handlers.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # -- Routers --
    app.include_router(health.router)
    app.include_router(users.router, prefix="/api/v1")
    app.include_router(items.router, prefix="/api/v1")

    # -- Middleware --
    # Note: last added = outermost = first to execute on requests.
    app.add_middleware(GZipMiddleware, minimum_size=500)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


app = create_app()
