---
name: package-managing
description: >
  Guides the agent through Python project management with uv, the fast Rust-based
  package and project manager. Triggered when users say "create a Python project",
  "init a Python project with uv", "add a dependency", "manage Python packages",
  "sync dependencies", "lock dependencies", "run a Python script", "set up pyproject.toml",
  or mention uv, package management, virtual environments, or Python project initialization.
version: 1.0.0
---

# uv Package and Project Manager

## Overview

uv is a fast Python package and project manager written in Rust. It serves as a single tool replacing pip, pip-tools, pipx, pyenv, virtualenv, and poetry. Use uv for dependency resolution, virtual environment management, Python version management, and project scaffolding.

Key characteristics:
- 10-100x faster than pip for dependency resolution and installation
- Built-in virtual environment management (automatic `.venv` creation)
- Lockfile support for reproducible builds
- Python version management without pyenv
- Cross-platform with no Python prerequisite for installation

## Project Initialization

### Creating a New Project

Use `uv init` to scaffold a new Python project.

```bash
# Create a project in a new directory
uv init my-project

# Initialize in the current directory
uv init

# Create an application project (default, no build system)
uv init --app my-app

# Create a library project (includes build system for publishing)
uv init --lib my-lib

# Specify minimum Python version at init time
uv init --python ">=3.12" my-project
```

### Project Layout

An application project (`--app`) produces:

```
my-app/
  .python-version
  pyproject.toml
  README.md
  main.py
```

A library project (`--lib`) produces:

```
my-lib/
  .python-version
  pyproject.toml
  README.md
  src/
    my_lib/
      __init__.py
      py.typed
```

For FastAPI services, prefer `--app` and reorganize into a `src/` layout manually if needed.

## pyproject.toml Structure

The `pyproject.toml` file is the single source of truth for project metadata, dependencies, and tool configuration.

### Core Sections

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "A short project description"
requires-python = ">=3.12"
readme = "README.md"
license = { text = "MIT" }
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Dependency Groups

Use `[dependency-groups]` to organize development and optional dependency sets.

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
    "ruff>=0.9",
    "mypy>=1.14",
]
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
]
```

### uv-Specific Configuration

The `[tool.uv]` section controls uv behavior.

```toml
[tool.uv]
# Override dependency resolution for specific packages
override-dependencies = [
    "numpy>=2.0",
]

# Constrain transitive dependency versions
constraint-dependencies = [
    "grpcio<1.70",
]

# Default groups to install on uv sync
default-groups = ["dev"]
```

See `assets/pyproject-template.toml` for a production-ready FastAPI project template.

## Dependency Management

### Adding Dependencies

```bash
# Add a production dependency
uv add fastapi

# Add with version constraint
uv add "pydantic>=2.0,<3.0"

# Add multiple packages at once
uv add fastapi uvicorn pydantic-settings

# Add a development dependency
uv add --dev pytest ruff mypy

# Add to a named dependency group
uv add --group docs mkdocs mkdocs-material

# Add with extras
uv add "uvicorn[standard]"

# Add a git dependency
uv add "my-package @ git+https://github.com/org/repo.git@main"

# Add a local path dependency (editable)
uv add --editable "../shared-lib"
```

### Removing Dependencies

```bash
# Remove a production dependency
uv remove httpx

# Remove a dev dependency
uv remove --dev mypy

# Remove from a named group
uv remove --group docs mkdocs
```

### Version Constraints

uv follows PEP 440 version specifiers:

| Specifier        | Meaning                              |
|------------------|--------------------------------------|
| `>=1.0`          | Version 1.0 or higher               |
| `>=1.0,<2.0`    | At least 1.0, below 2.0             |
| `~=1.4`         | Compatible release (>=1.4, <2.0)    |
| `==1.4.*`       | Any patch of 1.4                     |
| `>=1.0; python_version>="3.12"` | Environment marker     |

## Lock File

### Generating the Lock File

The `uv.lock` file pins every direct and transitive dependency to an exact version for reproducible installs.

```bash
# Generate or update the lock file
uv lock

# Update a single package in the lock file
uv lock --upgrade-package fastapi

# Upgrade all packages to latest compatible versions
uv lock --upgrade
```

### When to Lock vs Sync

- Run `uv lock` after editing `pyproject.toml` manually or to refresh resolved versions.
- Run `uv sync` to install dependencies from the lock file into the virtual environment.
- `uv add` and `uv remove` automatically update both `pyproject.toml` and `uv.lock`.

Always commit `uv.lock` to version control for applications. For libraries, committing the lock file is optional but recommended for CI reproducibility.

## Syncing the Environment

`uv sync` installs the exact versions from `uv.lock` into the project virtual environment.

```bash
# Sync all dependencies (creates .venv if missing)
uv sync

# Sync without updating the lock file (CI-friendly, fails if lock is stale)
uv sync --frozen

# Sync without dev dependencies (production builds)
uv sync --no-dev

# Sync including a specific extra group
uv sync --group docs

# Sync all groups
uv sync --all-groups

# Reinstall all packages from scratch
uv sync --reinstall
```

In CI pipelines, always use `uv sync --frozen` to ensure the lock file is up to date and avoid unexpected resolution changes.

## Running Commands

Use `uv run` to execute commands within the project virtual environment without manually activating it.

```bash
# Run a Python script
uv run python main.py

# Run pytest
uv run pytest tests/ -v

# Run a FastAPI dev server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run FastAPI CLI dev server
uv run fastapi dev app/main.py

# Run ruff linter
uv run ruff check .

# Run mypy type checker
uv run mypy src/

# Run an arbitrary module
uv run python -m my_module
```

`uv run` automatically creates the virtual environment and syncs dependencies if needed before executing the command.

## Virtual Environments

### Automatic Management

uv creates and manages a `.venv` directory in the project root automatically. There is no need to manually create or activate virtual environments for most workflows -- `uv run`, `uv sync`, and `uv add` all handle this.

### Manual Creation

```bash
# Create a venv with the default Python
uv venv

# Create a venv with a specific Python version
uv venv --python 3.12

# Create a venv at a custom path
uv venv /path/to/venv

# Activate manually (rarely needed with uv run)
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```

### IDE Integration

Point IDE Python interpreters to `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows) to get autocomplete and type checking aligned with project dependencies.

## Python Version Management

uv can install and manage Python versions directly, replacing pyenv.

```bash
# Install a specific Python version
uv python install 3.12

# Install multiple versions
uv python install 3.11 3.12 3.13

# Pin the project to a specific version (writes .python-version)
uv python pin 3.12

# List available Python versions
uv python list

# List installed Python versions
uv python list --only-installed
```

The `.python-version` file is read by uv automatically. Commit it to version control to ensure all contributors use the same Python version.

## Script Inline Dependencies

For standalone scripts that need third-party packages, use PEP 723 inline metadata instead of creating a full project.

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx>=0.27",
#     "rich>=13.0",
# ]
# ///

import httpx
from rich import print

resp = httpx.get("https://api.example.com/data")
print(resp.json())
```

Run the script directly with uv:

```bash
uv run script.py
```

uv resolves and installs the inline dependencies into an isolated, cached environment automatically. This is ideal for utility scripts, one-off tasks, and scripts shared outside a project context.

## Tool Management

uv replaces pipx for installing and running Python CLI tools globally.

```bash
# Install a tool globally
uv tool install ruff
uv tool install uv-publish

# Run a tool without installing (ephemeral)
uv tool run ruff check .
uv tool run --from "huggingface-hub" huggingface-cli

# Shorthand: uvx (alias for uv tool run)
uvx ruff check .
uvx black .

# List installed tools
uv tool list

# Upgrade a tool
uv tool upgrade ruff

# Uninstall a tool
uv tool uninstall ruff
```

Prefer `uv tool run` / `uvx` for infrequent use. Prefer `uv tool install` for tools used regularly across projects.

## Workspace Support

Workspaces allow managing multiple related packages in a monorepo with a shared lock file.

### Root pyproject.toml

```toml
[tool.uv.workspace]
members = [
    "packages/*",
    "services/*",
]
```

### Workspace Layout

```
monorepo/
  pyproject.toml          # Root workspace config
  uv.lock                 # Single shared lock file
  packages/
    shared-models/
      pyproject.toml
      src/shared_models/
    shared-utils/
      pyproject.toml
      src/shared_utils/
  services/
    api-service/
      pyproject.toml
      src/api_service/
    worker-service/
      pyproject.toml
      src/worker_service/
```

### Inter-Package Dependencies

Reference workspace members as path dependencies:

```toml
# In services/api-service/pyproject.toml
[project]
dependencies = [
    "shared-models",
    "shared-utils",
]

[tool.uv.sources]
shared-models = { workspace = true }
shared-utils = { workspace = true }
```

Run commands scoped to a specific workspace member:

```bash
# Sync a specific member
uv sync --package api-service

# Run tests for a specific member
uv run --package api-service pytest
```

## Common Workflows

### Starting a New FastAPI Project

```bash
uv init --app my-api
cd my-api
uv add fastapi "uvicorn[standard]" pydantic-settings
uv add --dev pytest pytest-asyncio ruff mypy
uv run fastapi dev main.py
```

### Adding and Pinning a Dependency

```bash
uv add "sqlalchemy>=2.0,<3.0"
uv lock
uv sync
```

### CI Pipeline

```bash
uv sync --frozen --no-dev
uv run pytest --tb=short
uv run ruff check .
uv run mypy src/
```

### Upgrading Dependencies

```bash
# Upgrade everything
uv lock --upgrade
uv sync

# Upgrade a single package
uv lock --upgrade-package fastapi
uv sync
```

## Cross-References

- For FastAPI project setup, combine with the `fastapi` skill.
- See `assets/pyproject-template.toml` for a complete production-ready project template.
