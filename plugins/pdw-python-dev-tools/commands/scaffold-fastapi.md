---
name: scaffold-fastapi
description: Scaffold a complete FastAPI project with uv, Pydantic models, async patterns, Docker, and optional LangChain/LangGraph integration
argument-hint: "project-name [--with-langchain] [--with-docker] [--with-db postgres|sqlite]"
disable-model-invocation: true
---

# Scaffold FastAPI Project

Scaffold a production-ready FastAPI project using the full Python stack.

## Workflow

### Phase 1: Gather Requirements

Ask the user for the following project details (skip if provided as arguments):

1. **Project name** (kebab-case, e.g., `my-api-service`)
2. **Python version** (default: 3.12)
3. **Features to include:**
   - Database integration (PostgreSQL / SQLite / None)
   - LangChain/LangGraph AI agent support
   - Docker configuration
   - Authentication (JWT / API Key / None)

### Phase 2: Initialize Project with uv

Consult the `package-managing` skill at `${CLAUDE_PLUGIN_ROOT}/skills/package-managing/SKILL.md` for project initialization patterns.

1. Run `uv init --app <project-name>`
2. Configure `pyproject.toml` based on the template at `${CLAUDE_PLUGIN_ROOT}/skills/package-managing/assets/pyproject-template.toml`
3. Add dependencies based on selected features:
   - **Core**: fastapi, uvicorn[standard], pydantic, pydantic-settings, httpx
   - **Database (PostgreSQL)**: sqlalchemy[asyncio], asyncpg, alembic
   - **Database (SQLite)**: sqlalchemy[asyncio], aiosqlite, alembic
   - **LangChain**: langchain, langchain-anthropic, langgraph, langchain-community
   - **Auth JWT**: python-jose[cryptography], passlib[bcrypt]
4. Add dev dependencies: pytest, pytest-asyncio, pytest-cov, ruff, mypy
5. Run `uv sync`

### Phase 3: Create Application Structure

Consult the `app-scaffolding` skill at `${CLAUDE_PLUGIN_ROOT}/skills/app-scaffolding/SKILL.md` for application patterns.

Use templates from `${CLAUDE_PLUGIN_ROOT}/skills/app-scaffolding/assets/app-template/` as the base.

Create the following structure:

```
<project-name>/
├── src/
│   └── <package_name>/
│       ├── __init__.py
│       ├── main.py              # App factory with lifespan
│       ├── config.py            # Pydantic Settings
│       ├── dependencies.py      # Common dependencies
│       ├── routers/
│       │   ├── __init__.py
│       │   └── health.py        # Health check router
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py       # Pydantic schemas
│       └── services/
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Async fixtures, TestClient
│   └── test_health.py           # Health endpoint test
├── pyproject.toml
├── .env.example
├── .gitignore
└── .python-version
```

### Phase 4: Create Pydantic Models

Consult the `pydantic` skill at `${CLAUDE_PLUGIN_ROOT}/skills/pydantic/SKILL.md` for model patterns.

- Create `config.py` with `BaseSettings` for environment configuration
- Create base schemas: `BaseSchema`, `TimestampMixin`, standard response models
- Create feature-specific schemas as needed

### Phase 5: Add Database Layer (if selected)

If database integration was selected:

- Create `db/` directory with engine, session, and base model
- Create async session dependency with yield pattern
- Set up Alembic for migrations: `uv run alembic init -t async alembic`
- Create initial migration

### Phase 6: Add LangChain/LangGraph (if selected)

Consult the `agent-workflow` skill at `${CLAUDE_PLUGIN_ROOT}/skills/agent-workflow/SKILL.md`.

Use the template at `${CLAUDE_PLUGIN_ROOT}/skills/agent-workflow/assets/graph-template.py`.

- Create `agents/` directory with agent graph definition
- Create tool definitions
- Add `/chat` streaming endpoint
- Wire agent into FastAPI via dependency injection

### Phase 7: Add Docker Configuration (if selected)

Consult the `docker-build` skill at `${CLAUDE_PLUGIN_ROOT}/skills/docker-build/SKILL.md`.

Copy and adapt templates from `${CLAUDE_PLUGIN_ROOT}/skills/docker-build/assets/`:

- `Dockerfile` (from Dockerfile.fastapi)
- `Dockerfile.dev` (for development)
- `docker-compose.yml` (adapt based on selected services)
- `.dockerignore`

### Phase 8: Validate

1. Run `uv sync` to verify dependencies resolve
2. Run `uv run uvicorn <package>.main:app --reload` to verify app starts
3. Run `uv run pytest` to verify tests pass
4. If Docker selected: run `docker compose build` to verify build

### Phase 9: Summary

Present the user with:
- Project structure overview
- Available commands (run, test, lint, format, Docker)
- Next steps recommendations
