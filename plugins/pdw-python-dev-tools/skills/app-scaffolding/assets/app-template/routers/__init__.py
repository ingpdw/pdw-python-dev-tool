"""Router package.

Each module in this package defines an `APIRouter` instance named `router`.
Import and include them in the application factory (`main.py`):

    from app.routers import health, users, items

    app.include_router(health.router)
    app.include_router(users.router, prefix="/api/v1")
    app.include_router(items.router, prefix="/api/v1")

To add a new router:
1. Create a new module (e.g., `orders.py`) in this package.
2. Define `router = APIRouter(prefix="/orders", tags=["orders"])`.
3. Add route handlers to the router.
4. Import and include it in `main.py`.
"""
