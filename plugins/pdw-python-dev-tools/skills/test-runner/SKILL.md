---
name: test-runner
description: >
  Guides the agent through running and writing Python tests with pytest. Triggered when
  users say "run tests", "write a test", "test this function", "add unit tests",
  "run pytest", "check test coverage", "debug failing test", "create test fixtures",
  "mock a dependency", or mention pytest, pytest-asyncio, pytest-cov, testing,
  unit tests, integration tests, test coverage, or test-driven development.
version: 1.0.0
---

# Python Test Runner

## Overview

Use **pytest** as the primary test framework for all Python projects. Combine with **pytest-asyncio** for async test support, **pytest-cov** for coverage reporting, and **httpx** (`AsyncClient`) for FastAPI endpoint testing.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_users.py

# Run a specific test function
uv run pytest tests/test_users.py::test_create_user

# Run tests matching a keyword expression
uv run pytest -k "create or delete"

# Run tests by marker
uv run pytest -m "slow"
```

### Coverage Reporting

```bash
# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html

# Fail if coverage is below threshold
uv run pytest --cov=src --cov-fail-under=80
```

### Useful Flags

```bash
# Stop on first failure
uv run pytest -x

# Stop after N failures
uv run pytest --maxfail=3

# Show local variables in traceback
uv run pytest -l

# Run last failed tests first
uv run pytest --lf

# Run only tests that failed last time
uv run pytest --ff

# Parallel execution (requires pytest-xdist)
uv run pytest -n auto

# Show slowest N tests
uv run pytest --durations=10

# Disable output capture (show print statements)
uv run pytest -s
```

## pytest Configuration

### pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
python_files = ["test_*.py"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
]
```

## Writing Tests

### Basic Test Structure

```python
import pytest


def test_addition():
    assert 1 + 1 == 2


def test_exception_raised():
    with pytest.raises(ValueError, match="invalid value"):
        raise ValueError("invalid value")


class TestUserService:
    def test_create_user(self):
        user = create_user(name="Alice")
        assert user.name == "Alice"

    def test_create_user_empty_name(self):
        with pytest.raises(ValueError):
            create_user(name="")
```

### Fixtures

```python
import pytest


@pytest.fixture
def sample_user():
    return {"name": "Alice", "email": "alice@example.com"}


@pytest.fixture
def db_session():
    session = create_session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(autouse=True)
def reset_cache():
    """Automatically reset cache before each test."""
    cache.clear()
    yield
    cache.clear()


# Fixture with parameters
@pytest.fixture(params=["sqlite", "postgres"])
def database(request):
    db = create_database(request.param)
    yield db
    db.drop()
```

### conftest.py

Place shared fixtures in `conftest.py` at the appropriate directory level:

```python
# tests/conftest.py
import pytest
from httpx import ASGITransport, AsyncClient

from myapp.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

### Async Tests

With `asyncio_mode = "auto"` in pyproject.toml, async test functions are automatically recognized:

```python
async def test_async_endpoint(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_async_service():
    result = await fetch_data("https://api.example.com/data")
    assert result is not None
```

### FastAPI Endpoint Testing

```python
import pytest
from httpx import ASGITransport, AsyncClient

from myapp.main import create_app
from myapp.dependencies import get_db


@pytest.fixture
def app():
    app = create_app()
    # Override dependencies for testing
    app.dependency_overrides[get_db] = lambda: fake_db
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_create_item(client):
    response = await client.post(
        "/api/items",
        json={"name": "Test Item", "price": 9.99},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Item"


async def test_get_item_not_found(client):
    response = await client.get("/api/items/999")
    assert response.status_code == 404
```

### Mocking

```python
from unittest.mock import AsyncMock, MagicMock, patch


def test_with_mock():
    mock_service = MagicMock()
    mock_service.get_user.return_value = {"name": "Alice"}

    result = process_user(mock_service, user_id=1)
    mock_service.get_user.assert_called_once_with(1)
    assert result["name"] == "Alice"


async def test_with_async_mock():
    mock_client = AsyncMock()
    mock_client.fetch.return_value = {"data": "value"}

    result = await handler(mock_client)
    mock_client.fetch.assert_awaited_once()


def test_with_patch():
    with patch("myapp.services.external_api.call") as mock_call:
        mock_call.return_value = {"status": "ok"}
        result = my_function()
        assert result["status"] == "ok"


# Patch as decorator
@patch("myapp.services.send_email")
def test_send_notification(mock_send):
    mock_send.return_value = True
    notify_user(user_id=1)
    mock_send.assert_called_once()
```

### Parametrize

```python
import pytest


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (1, 2),
        (2, 4),
        (3, 6),
        (-1, -2),
    ],
)
def test_double(input_val, expected):
    assert double(input_val) == expected


@pytest.mark.parametrize(
    "status_code, is_success",
    [
        (200, True),
        (201, True),
        (400, False),
        (500, False),
    ],
    ids=["ok", "created", "bad-request", "server-error"],
)
def test_is_success_status(status_code, is_success):
    assert is_success_response(status_code) == is_success
```

### Markers

```python
import pytest


@pytest.mark.slow
def test_large_dataset():
    """Takes a long time to run."""
    ...


@pytest.mark.integration
async def test_database_connection():
    """Requires a running database."""
    ...


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-only test",
)
def test_unix_socket():
    ...


@pytest.mark.xfail(reason="Known bug #123")
def test_known_bug():
    ...
```

## Test Project Structure

```
project/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── main.py
│       ├── services/
│       └── models/
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures (app, client, db)
│   ├── test_health.py        # Health endpoint tests
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   └── test_services.py
│   └── integration/
│       ├── __init__.py
│       └── test_api.py
└── pyproject.toml
```

## Fixture Scope

Control how often a fixture is created with the `scope` parameter:

```python
@pytest.fixture(scope="function")  # default: new instance per test
def db_session():
    ...

@pytest.fixture(scope="module")    # shared across tests in one module
def db_engine():
    ...

@pytest.fixture(scope="session")   # shared across the entire test session
def app():
    return create_app()
```

| Scope      | Lifetime                          | Use Case                                |
|------------|-----------------------------------|-----------------------------------------|
| `function` | One per test function (default)   | Mutable state, per-test isolation       |
| `class`    | One per test class                | Shared setup within a test class        |
| `module`   | One per test module               | Expensive resources (DB engine, client) |
| `session`  | One per entire test session       | Immutable globals, app factory          |

Use the narrowest scope that avoids unnecessary recreation. Async fixtures follow the same scoping rules with `pytest-asyncio`.

## Dependency Override Cleanup

When overriding FastAPI dependencies in tests, always clean up to prevent test pollution:

```python
@pytest.fixture
def app():
    app = create_app()
    app.dependency_overrides[get_db] = lambda: fake_db
    yield app
    app.dependency_overrides.clear()  # CRITICAL: prevent leaking to next test
```

Alternatively, use `autouse` fixtures for automatic cleanup:

```python
@pytest.fixture(autouse=True)
def _cleanup_overrides(app):
    yield
    app.dependency_overrides.clear()
```

## Debugging Tests

```bash
# Drop into debugger on failure
uv run pytest --pdb

# Drop into debugger at first line of each test
uv run pytest --trace

# Show full diff for assertion errors
uv run pytest -vv

# Enable asyncio debug mode
PYTHONASYNCIODEBUG=1 uv run pytest
```

## Best Practices

1. **Name tests clearly**: `test_<what>_<condition>_<expected>` (e.g., `test_create_user_empty_name_raises_error`)
2. **One assertion per test** when possible — makes failures easier to diagnose
3. **Use fixtures** for setup/teardown instead of `setUp`/`tearDown` methods
4. **Test behavior, not implementation** — assert on outputs, not internal state
5. **Use `conftest.py`** for shared fixtures instead of base test classes
6. **Keep tests fast** — mock external services, use in-memory databases for unit tests
7. **Use `pytest.raises`** context manager for exception assertions
8. **Use `parametrize`** to avoid duplicating test logic
9. **Run coverage** regularly — aim for meaningful coverage, not 100%
10. **Separate unit and integration tests** with markers or directories
