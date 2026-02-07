---
name: pydantic
description: >
  Activated when the user wants to create a data model, validate data, serialize JSON,
  create Pydantic models, add validators, define settings, or create request/response schemas.
  Covers Pydantic v2 BaseModel, Field, validators, data validation, JSON schema generation,
  serialization, deserialization, and settings management.
version: 1.0.0
---

# Pydantic v2

Pydantic v2 is the standard library for data validation, serialization, and settings management in Python. It uses type annotations to define data schemas and validates data at runtime with a high-performance Rust core.

## Model Basics

Define models by subclassing `BaseModel`. Use standard type annotations for fields and `Field()` for metadata and constraints.

```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class User(BaseModel):
    id: int
    name: str = Field(min_length=1, max_length=100, description="Full name")
    email: str
    age: Optional[int] = Field(default=None, ge=0, le=150)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Field Basics

- **Required fields**: declare with a type annotation and no default.
- **Optional fields**: use `Optional[T]` (or `T | None`) with a default of `None`.
- **Default values**: assign a literal or use `Field(default=...)`.
- **Default factories**: use `Field(default_factory=callable)` for mutable defaults.

### Common `Field()` Parameters

| Parameter       | Purpose                              |
|-----------------|--------------------------------------|
| `default`       | Static default value                 |
| `default_factory` | Callable producing a default      |
| `alias`         | Alternative name for parsing input   |
| `title`         | Human-readable title for JSON Schema |
| `description`   | Field description for JSON Schema    |
| `ge`, `gt`, `le`, `lt` | Numeric constraints           |
| `min_length`, `max_length` | String/collection length  |
| `pattern`       | Regex pattern for strings            |
| `exclude`       | Exclude from serialization           |
| `frozen`        | Make individual field immutable       |

## Common Field Types and Constraints

Pydantic ships constrained types and special types for common patterns.

```python
from pydantic import (
    BaseModel,
    EmailStr,
    HttpUrl,
    Field,
    constr,
    conint,
    confloat,
    conlist,
)


class Profile(BaseModel):
    username: constr(min_length=3, max_length=30, pattern=r"^[a-zA-Z0-9_]+$")
    email: EmailStr
    website: HttpUrl | None = None
    score: conint(ge=0, le=100) = 0
    rating: confloat(ge=0.0, le=5.0) = 0.0
    tags: conlist(str, min_length=1, max_length=10) = ["general"]
```

> Install `pydantic[email]` to use `EmailStr`.

### Frequently Used Types

- `EmailStr` -- validated email address.
- `HttpUrl`, `AnyUrl` -- validated URLs.
- `IPvAnyAddress` -- IPv4 or IPv6 address.
- `SecretStr` -- string hidden in repr and serialization.
- `FilePath`, `DirectoryPath` -- validated filesystem paths.
- `PastDatetime`, `FutureDatetime` -- temporal constraints.
- `PositiveInt`, `NegativeInt`, `NonNegativeInt` -- numeric shortcuts.
- `UUID4` -- UUID version 4.
- `Json[T]` -- parse a JSON string into type `T`.

## Model Configuration

Control model behavior with `model_config` using `ConfigDict`.

```python
from pydantic import BaseModel, ConfigDict


class Item(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=False,
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
        validate_default=True,
        strict=False,
        json_schema_extra={"examples": [{"item_name": "Widget", "price": 9.99}]},
    )

    item_name: str = Field(alias="itemName")
    price: float
```

### Key Configuration Options

| Option                  | Effect                                                    |
|-------------------------|-----------------------------------------------------------|
| `str_strip_whitespace`  | Strip leading/trailing whitespace from strings            |
| `frozen`                | Make all fields immutable (model becomes hashable)        |
| `from_attributes`       | Allow initialization from ORM objects via attribute access|
| `populate_by_name`      | Accept both field name and alias during parsing           |
| `use_enum_values`       | Store enum values instead of enum members                 |
| `validate_default`      | Run validators on default values                          |
| `strict`                | Disable type coercion globally                            |
| `extra`                 | `"forbid"`, `"allow"`, or `"ignore"` extra fields         |
| `json_schema_extra`     | Extend generated JSON Schema                              |

## Validators

### Field Validators

Apply validation logic to individual fields with `@field_validator`.

```python
from pydantic import BaseModel, field_validator


class Signup(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_must_be_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("must be alphanumeric")
        return v.lower()

    @field_validator("password", mode="after")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("must be at least 8 characters")
        return v
```

- `mode="before"` -- run before Pydantic's own parsing (receives raw input).
- `mode="after"` (default) -- run after Pydantic parses and coerces the value.
- Apply to multiple fields: `@field_validator("field_a", "field_b")`.

### Model Validators

Validate the entire model for cross-field logic with `@model_validator`.

```python
from pydantic import BaseModel, model_validator


class DateRange(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def check_date_order(self) -> "DateRange":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self
```

- `mode="before"` -- receives raw input dict; useful for restructuring data.
- `mode="after"` -- receives the fully constructed model instance.

For advanced validator patterns including `BeforeValidator`, `AfterValidator`, `WrapValidator`, custom types, and discriminated unions, consult the [validators reference](references/validators.md).

## Computed Fields

Derive fields from other values without storing them. Computed fields appear in serialization and JSON Schema.

```python
from pydantic import BaseModel, computed_field


class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height
```

## Serialization

### `model_dump()`

Convert a model instance to a dictionary.

```python
user = User(id=1, name="Ada", email="ada@example.com")
data = user.model_dump()                        # full dict
data = user.model_dump(exclude_none=True)        # drop None values
data = user.model_dump(include={"id", "name"})   # only selected fields
data = user.model_dump(exclude={"created_at"})   # exclude fields
data = user.model_dump(by_alias=True)            # use alias names as keys
data = user.model_dump(mode="json")              # JSON-compatible types
```

### `model_dump_json()`

Serialize directly to a JSON string (faster than `json.dumps(model.model_dump())`).

```python
json_str = user.model_dump_json(indent=2, exclude_none=True)
```

### Serialization Aliases

Use `serialization_alias` to control output key names independently from parsing aliases.

```python
class ApiResponse(BaseModel):
    internal_id: int = Field(serialization_alias="id")
    status_code: int = Field(serialization_alias="status")
```

### Custom Serializers

```python
from pydantic import field_serializer

class Event(BaseModel):
    timestamp: datetime

    @field_serializer("timestamp")
    def serialize_ts(self, v: datetime, _info) -> str:
        return v.isoformat()
```

## Deserialization

### From a Dictionary

```python
user = User.model_validate({"id": 1, "name": "Ada", "email": "ada@example.com"})
```

### From a JSON String

```python
user = User.model_validate_json('{"id": 1, "name": "Ada", "email": "ada@example.com"}')
```

### From ORM / Arbitrary Objects

With `from_attributes=True` in `model_config`, initialize from objects with matching attributes.

```python
class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str

# sqlalchemy_user is an ORM instance
user_read = UserRead.model_validate(sqlalchemy_user)
```

## Settings Management

Use `BaseSettings` to load configuration from environment variables, `.env` files, and other sources.

```bash
pip install pydantic-settings
```

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    app_name: str = "My App"
    debug: bool = False
    database_url: str
    redis_url: str = "redis://localhost:6379"
    allowed_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
```

### Environment Variable Mapping

With `env_prefix="APP_"`, the field `database_url` maps to `APP_DATABASE_URL`.

### Nested Settings

Use `env_nested_delimiter` to map nested models from flat environment variables.

```python
class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "mydb"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    db: DatabaseSettings = DatabaseSettings()
```

Set `DB__HOST=prod-db` and `DB__PORT=5433` in the environment to override nested values.

### Secrets Directory

Load secrets from files (e.g., Docker secrets):

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(secrets_dir="/run/secrets")

    db_password: str
```

## Model Inheritance and Composition

### Inheritance

Share common fields across models with standard class inheritance.

```python
class TimestampMixin(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None


class UserCreate(TimestampMixin):
    name: str
    email: str


class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
```

### Composition (Nested Models)

Nest models for structured data. Pydantic validates nested objects recursively.

```python
class Address(BaseModel):
    street: str
    city: str
    country: str = "US"


class Company(BaseModel):
    name: str
    address: Address
    employees: list["Employee"] = []


class Employee(BaseModel):
    name: str
    role: str

Company.model_rebuild()  # resolve forward references
```

## Generic Models

Create reusable, type-parameterized models with `Generic[T]`.

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    per_page: int
    has_next: bool

    @computed_field
    @property
    def total_pages(self) -> int:
        return (self.total + self.per_page - 1) // self.per_page


# Concrete usage
class UserList(PaginatedResponse[User]):
    pass

# Or inline
response = PaginatedResponse[User].model_validate(data)
```

Generic models generate accurate JSON Schemas per specialization, making them ideal for API response wrappers.

## JSON Schema Generation

Generate JSON Schema from any model for documentation, API specs, or frontend validation.

```python
schema = User.model_json_schema()
```

Customize with `json_schema_extra` in `model_config` or `Field()`:

```python
class Product(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"name": "Widget", "price": 9.99}]
        }
    )
    name: str
    price: float = Field(json_schema_extra={"examples": [9.99, 19.99]})
```

---

For FastAPI request/response models, consult the `fastapi` skill.

For advanced validator patterns, discriminated unions, recursive models, and custom types, consult the [validators reference](references/validators.md).
