# Pydantic Validators Reference

Comprehensive reference for Pydantic v2 validation patterns, custom types, discriminated unions, recursive models, and error handling.

## @field_validator Deep Dive

### mode="before" vs mode="after"

A `before` validator receives the raw input value before Pydantic performs type parsing or coercion. An `after` validator receives the already-parsed and typed value.

```python
from pydantic import BaseModel, field_validator


class Product(BaseModel):
    price: float
    tags: list[str]

    @field_validator("price", mode="before")
    @classmethod
    def parse_price(cls, v):
        """Accept strings like '$19.99' and strip the currency symbol."""
        if isinstance(v, str):
            return float(v.replace("$", "").replace(",", ""))
        return v

    @field_validator("tags", mode="after")
    @classmethod
    def lowercase_tags(cls, v: list[str]) -> list[str]:
        """Normalize tags after list parsing."""
        return [tag.lower().strip() for tag in v]
```

**When to use `mode="before"`:**
- Accept multiple input formats (string, int, dict) and normalize them.
- Pre-process raw data before Pydantic's type coercion.
- Handle legacy or external data formats.

**When to use `mode="after"` (default):**
- The value is already the correct type; apply business rules.
- Transform or constrain a typed value (clamping, normalization).

### Validating Multiple Fields

Apply one validator to several fields by listing them as positional arguments.

```python
class Form(BaseModel):
    first_name: str
    last_name: str
    city: str

    @field_validator("first_name", "last_name", "city")
    @classmethod
    def must_be_title_case(cls, v: str) -> str:
        return v.strip().title()
```

Use `"*"` to apply a validator to every field:

```python
    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v
```

### Reusable Validators

Extract common logic into plain functions and reference them from multiple models.

```python
from pydantic import field_validator


def normalize_whitespace(v: str) -> str:
    """Collapse consecutive whitespace and strip."""
    import re
    return re.sub(r"\s+", " ", v).strip()


class Author(BaseModel):
    name: str

    normalize_name = field_validator("name")(classmethod(lambda cls, v: normalize_whitespace(v)))


class Publisher(BaseModel):
    name: str

    normalize_name = field_validator("name")(classmethod(lambda cls, v: normalize_whitespace(v)))
```

A cleaner pattern uses `Annotated` types (see the Annotated Validators section below).

### Accessing Other Field Values

In `mode="after"`, use `info.data` to read previously validated fields (fields are validated in definition order).

```python
from pydantic import BaseModel, field_validator, ValidationInfo


class Discount(BaseModel):
    original_price: float
    discount_percent: float
    final_price: float

    @field_validator("final_price", mode="after")
    @classmethod
    def check_final_price(cls, v: float, info: ValidationInfo) -> float:
        expected = info.data["original_price"] * (1 - info.data["discount_percent"] / 100)
        if abs(v - expected) > 0.01:
            raise ValueError(f"final_price should be {expected:.2f}")
        return round(v, 2)
```

## @model_validator

### mode="before"

Receives the raw input (typically a `dict`) before any field parsing. Return the modified dict.

```python
from pydantic import BaseModel, model_validator
from typing import Any


class LegacyUser(BaseModel):
    full_name: str
    email: str

    @model_validator(mode="before")
    @classmethod
    def flatten_legacy_format(cls, data: Any) -> Any:
        """Handle legacy API format with nested 'info' key."""
        if isinstance(data, dict) and "info" in data:
            info = data.pop("info")
            data.update(info)
        return data
```

### mode="after"

Receives the fully constructed model instance. Return `self` after validation.

```python
class Reservation(BaseModel):
    check_in: date
    check_out: date
    guests: int
    rooms: int

    @model_validator(mode="after")
    def validate_reservation(self) -> "Reservation":
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        if self.guests > self.rooms * 4:
            raise ValueError("too many guests for the number of rooms")
        return self
```

### mode="wrap"

Wrap validators receive both the input and a handler function, providing full control over the validation pipeline.

```python
from pydantic import BaseModel, model_validator, ValidatorFunctionWrapHandler


class FlexibleModel(BaseModel):
    value: int

    @model_validator(mode="wrap")
    @classmethod
    def try_parse(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> "FlexibleModel":
        if isinstance(data, int):
            data = {"value": data}
        return handler(data)
```

## Annotated Validators

Define validators inline using `Annotated` for maximum reusability. These are type-level validators that travel with the type annotation.

### BeforeValidator

```python
from typing import Annotated
from pydantic import BaseModel, BeforeValidator


def strip_and_lower(v: str) -> str:
    if isinstance(v, str):
        return v.strip().lower()
    return v


CleanStr = Annotated[str, BeforeValidator(strip_and_lower)]


class Tag(BaseModel):
    name: CleanStr
    category: CleanStr
```

### AfterValidator

```python
from pydantic import AfterValidator


def must_be_positive(v: float) -> float:
    if v <= 0:
        raise ValueError("must be positive")
    return v


PositivePrice = Annotated[float, AfterValidator(must_be_positive)]


class LineItem(BaseModel):
    description: str
    price: PositivePrice
    quantity: Annotated[int, AfterValidator(lambda v: max(1, v))]
```

### PlainValidator

Replace Pydantic's entire parsing logic for a field. The validator alone is responsible for producing the final value.

```python
from pydantic import PlainValidator


def parse_bool_flexible(v: Any) -> bool:
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "on")
    return bool(v)


FlexBool = Annotated[bool, PlainValidator(parse_bool_flexible)]
```

### WrapValidator

Intercept the validation pipeline with access to the inner handler. Useful for logging, error transformation, or conditional bypass.

```python
from pydantic import WrapValidator
from pydantic import ValidationError


def coerce_none_to_default(v: Any, handler, info) -> int:
    """Treat None as 0 for this field."""
    if v is None:
        return 0
    return handler(v)


SafeInt = Annotated[int, WrapValidator(coerce_none_to_default)]


class Metrics(BaseModel):
    clicks: SafeInt = 0
    impressions: SafeInt = 0
```

### Stacking Validators

Multiple validators on the same type execute in order (outermost `Annotated` entry runs first for `Before`; innermost runs first for `After`).

```python
StrictCleanStr = Annotated[
    str,
    BeforeValidator(lambda v: v.strip() if isinstance(v, str) else v),
    AfterValidator(lambda v: v if len(v) > 0 else (_ for _ in ()).throw(ValueError("empty"))),
]
```

## Custom Types with __get_pydantic_core_schema__

Build fully custom types that integrate with Pydantic's core validation.

```python
from typing import Any
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class Color:
    def __init__(self, value: str):
        if not value.startswith("#") or len(value) != 7:
            raise ValueError("must be a hex color like #ff00aa")
        self.value = value

    def __repr__(self) -> str:
        return f"Color({self.value!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def _validate(cls, v: Any) -> "Color":
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            return cls(v)
        raise ValueError("string required")


class Theme(BaseModel):
    primary: Color
    secondary: Color
```

## Discriminated Unions (Tagged Unions)

Efficiently parse polymorphic data by inspecting a discriminator field to choose the correct model.

### String Discriminator

```python
from typing import Literal, Union
from pydantic import BaseModel, Field


class CreditCard(BaseModel):
    payment_type: Literal["credit_card"]
    card_number: str
    cvv: str


class BankTransfer(BaseModel):
    payment_type: Literal["bank_transfer"]
    account_number: str
    routing_number: str


class PayPal(BaseModel):
    payment_type: Literal["paypal"]
    email: str


class Order(BaseModel):
    id: int
    payment: CreditCard | BankTransfer | PayPal = Field(discriminator="payment_type")
```

### Callable Discriminator

Use `Discriminator()` with a callable for dynamic dispatch logic.

```python
from pydantic import Discriminator, Tag
from typing import Annotated, Union


def get_shape_type(data: dict) -> str:
    if "radius" in data:
        return "circle"
    if "width" in data:
        return "rectangle"
    return "unknown"


class Circle(BaseModel):
    radius: float


class Rectangle(BaseModel):
    width: float
    height: float


Shape = Annotated[
    Annotated[Circle, Tag("circle")] | Annotated[Rectangle, Tag("rectangle")],
    Discriminator(get_shape_type),
]


class Canvas(BaseModel):
    shapes: list[Shape]
```

### Performance Benefit

Discriminated unions skip trying each union member sequentially. Pydantic reads the discriminator field first and validates against only the matching model, providing O(1) dispatch instead of O(n).

## Recursive and Self-Referencing Models

### Direct Self-Reference

```python
class TreeNode(BaseModel):
    value: str
    children: list["TreeNode"] = []

TreeNode.model_rebuild()
```

Always call `model_rebuild()` after defining recursive models to resolve forward references.

### Mutual Recursion

```python
class Department(BaseModel):
    name: str
    manager: "Employee | None" = None
    sub_departments: list["Department"] = []


class Employee(BaseModel):
    name: str
    department: Department | None = None

Department.model_rebuild()
Employee.model_rebuild()
```

### Limiting Recursion Depth

Pydantic does not natively cap recursion depth. Use a model validator to enforce limits.

```python
class TreeNode(BaseModel):
    value: str
    children: list["TreeNode"] = []

    @model_validator(mode="after")
    def check_depth(self) -> "TreeNode":
        def _depth(node: "TreeNode") -> int:
            if not node.children:
                return 1
            return 1 + max(_depth(c) for c in node.children)

        if _depth(self) > 10:
            raise ValueError("tree exceeds maximum depth of 10")
        return self

TreeNode.model_rebuild()
```

## JSON Schema Customization

### Model-Level Customization

```python
class Sensor(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "IoT Sensor Reading",
            "description": "A single sensor data point",
            "examples": [
                {"sensor_id": "temp-01", "value": 22.5, "unit": "celsius"}
            ],
        }
    )

    sensor_id: str
    value: float
    unit: str
```

### Field-Level Customization

```python
class Coordinates(BaseModel):
    lat: float = Field(
        ge=-90,
        le=90,
        json_schema_extra={"examples": [37.7749]},
    )
    lon: float = Field(
        ge=-180,
        le=180,
        json_schema_extra={"examples": [-122.4194]},
    )
```

### Custom Schema Generator

Override schema generation for specific types.

```python
from pydantic import TypeAdapter

adapter = TypeAdapter(list[int])
schema = adapter.json_schema(
    ref_template="#/$defs/{model}",
    mode="serialization",
)
```

### Controlling Schema Mode

`model_json_schema(mode="validation")` generates the schema for input/parsing. `model_json_schema(mode="serialization")` generates the schema reflecting the serialized output, including computed fields and serialization aliases.

## Performance Tips

### Avoid Expensive Operations in Validators

Validators run on every instantiation. Keep them lightweight.

```python
# BAD: database query inside a validator
@field_validator("email")
@classmethod
def check_email_unique(cls, v):
    if db.query(User).filter_by(email=v).first():  # slow
        raise ValueError("email taken")
    return v

# GOOD: validate format only; check uniqueness in the service layer
@field_validator("email")
@classmethod
def normalize_email(cls, v: str) -> str:
    return v.strip().lower()
```

### Use model_rebuild() Correctly

Call `model_rebuild()` once after all forward-referencing models are defined. Do not call it inside loops or request handlers.

### TypeAdapter for Standalone Validation

When validating data without a full model (e.g., a plain list), use `TypeAdapter` instead of creating a wrapper model.

```python
from pydantic import TypeAdapter

ListOfInts = TypeAdapter(list[int])
result = ListOfInts.validate_python(["1", "2", "3"])  # [1, 2, 3]
json_bytes = ListOfInts.dump_json(result)
```

### Prefer Annotated Validators over Method Validators

Annotated validators are resolved at schema build time and stored efficiently in the core schema. They also promote reuse without inheritance.

### Frozen Models for Hashability

Setting `frozen=True` makes model instances immutable and hashable, enabling use as dict keys and set members. This can also prevent accidental mutation bugs.

```python
class CacheKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    namespace: str
    key: str
```

## Common Validation Patterns

### Email Normalization

```python
from pydantic import AfterValidator
from typing import Annotated


def normalize_email(v: str) -> str:
    local, domain = v.rsplit("@", 1)
    return f"{local.lower()}@{domain.lower()}"


NormalizedEmail = Annotated[EmailStr, AfterValidator(normalize_email)]
```

### Phone Number Formatting

```python
import re
from pydantic import BeforeValidator


def normalize_phone(v: str) -> str:
    digits = re.sub(r"\D", "", v)
    if len(digits) == 10:
        digits = "1" + digits
    if len(digits) != 11 or not digits.startswith("1"):
        raise ValueError("invalid US phone number")
    return f"+{digits[0]} ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"


USPhone = Annotated[str, BeforeValidator(normalize_phone)]
```

### Date Range Validation

```python
from datetime import date
from pydantic import model_validator


class DateRange(BaseModel):
    start_date: date
    end_date: date
    max_days: int = 365

    @model_validator(mode="after")
    def validate_range(self) -> "DateRange":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        delta = (self.end_date - self.start_date).days
        if delta > self.max_days:
            raise ValueError(f"date range must not exceed {self.max_days} days")
        return self
```

### Enum Coercion

Accept both enum members and their string/int values.

```python
from enum import Enum
from pydantic import BeforeValidator


class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


def coerce_status(v: Any) -> Status:
    if isinstance(v, Status):
        return v
    if isinstance(v, str):
        try:
            return Status(v.lower())
        except ValueError:
            pass
        try:
            return Status[v.upper()]
        except KeyError:
            pass
    raise ValueError(f"invalid status: {v}")


FlexStatus = Annotated[Status, BeforeValidator(coerce_status)]
```

### Slug Generation

```python
import re
from pydantic import model_validator


class Article(BaseModel):
    title: str
    slug: str | None = None

    @model_validator(mode="after")
    def generate_slug(self) -> "Article":
        if self.slug is None:
            slug = self.title.lower()
            slug = re.sub(r"[^\w\s-]", "", slug)
            slug = re.sub(r"[\s_]+", "-", slug).strip("-")
            self.slug = slug
        return self
```

## Error Handling

### Catching ValidationError

```python
from pydantic import BaseModel, ValidationError


class User(BaseModel):
    name: str
    age: int


try:
    User(name="", age="not_a_number")
except ValidationError as e:
    print(e.error_count())       # number of errors
    print(e.errors())            # list of error dicts
    print(e.json(indent=2))      # JSON representation
```

### Error Structure

Each error dict contains:

```python
{
    "type": "int_parsing",                 # error type identifier
    "loc": ("age",),                       # field location tuple
    "msg": "Input should be a valid integer, unable to parse string as an integer",
    "input": "not_a_number",               # the offending input
    "url": "https://errors.pydantic.dev/..."  # documentation link
}
```

### Custom Error Messages

```python
from pydantic import field_validator


class Registration(BaseModel):
    age: int

    @field_validator("age")
    @classmethod
    def check_age(cls, v: int) -> int:
        if v < 18:
            raise ValueError("registrants must be at least 18 years old")
        if v > 120:
            raise ValueError("please enter a realistic age")
        return v
```

### PydanticCustomError for Typed Errors

Create errors with custom types for programmatic error handling downstream.

```python
from pydantic_core import PydanticCustomError
from pydantic import field_validator


class Account(BaseModel):
    balance: float

    @field_validator("balance")
    @classmethod
    def check_balance(cls, v: float) -> float:
        if v < 0:
            raise PydanticCustomError(
                "negative_balance",
                "Account balance must not be negative, got {balance}",
                {"balance": v},
            )
        return v
```

The first argument is the error `type`, the second is a human-readable message template, and the third is a context dict interpolated into the template.

### Collecting Multiple Errors

Use `ValidationError` naturally -- Pydantic collects all field errors in a single pass and reports them together. There is no need for special configuration.

```python
try:
    User(name=123, age="abc")
except ValidationError as e:
    for error in e.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        print(f"{field}: {error['msg']}")
```

### Error Handling in APIs

When using Pydantic with FastAPI, `ValidationError` is automatically caught and returned as a 422 response. For non-FastAPI contexts, convert errors to a client-friendly format.

```python
def format_errors(exc: ValidationError) -> dict:
    return {
        "detail": [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
    }
```
