---
name: auth-security
description: >
  Guides the agent through implementing authentication and authorization in FastAPI
  applications. Triggered when users ask to "add authentication", "implement login",
  "add JWT tokens", "create OAuth2 flow", "hash passwords", "protect endpoints",
  "add role-based access", "implement RBAC", "add API key auth", "secure the API",
  or mention authentication, authorization, JWT, OAuth2, password hashing, bcrypt,
  access tokens, refresh tokens, security dependencies, or API security.
version: 1.0.0
---

# Authentication & Authorization for FastAPI

## Overview

FastAPI provides built-in security utilities based on OpenAPI standards. Use **OAuth2 with Password flow + JWT tokens** as the standard pattern for API authentication. Combine with **bcrypt** for password hashing and role-based access control (RBAC) for authorization.

Key packages:

```bash
uv add "python-jose[cryptography]" passlib[bcrypt] python-multipart
# or with PyJWT instead of python-jose:
uv add PyJWT[crypto] passlib[bcrypt] python-multipart
```

- `python-jose` or `PyJWT` -- JWT token creation and verification
- `passlib[bcrypt]` -- secure password hashing
- `python-multipart` -- required for OAuth2 form data parsing

## Password Hashing

Never store plaintext passwords. Use bcrypt through passlib:

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

## JWT Token Management

### Token Creation

```python
from datetime import datetime, timedelta, timezone

from jose import jwt  # or: import jwt (PyJWT)
from pydantic import BaseModel


class TokenConfig(BaseModel):
    secret_key: str = "your-secret-key"  # Use env variable in production
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7


token_config = TokenConfig()


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=token_config.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(
        to_encode,
        token_config.secret_key,
        algorithm=token_config.algorithm,
    )


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        days=token_config.refresh_token_expire_days
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(
        to_encode,
        token_config.secret_key,
        algorithm=token_config.algorithm,
    )
```

### Token Verification

```python
from jose import JWTError, jwt
from fastapi import HTTPException, status


def verify_token(token: str, expected_type: str = "access") -> dict:
    try:
        payload = jwt.decode(
            token,
            token_config.secret_key,
            algorithms=[token_config.algorithm],
        )
        if payload.get("type") != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
```

## FastAPI Security Dependencies

### OAuth2 Password Bearer

```python
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = verify_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    user = await db.get(User, int(user_id))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user
```

### Login Endpoint

```python
from fastapi import APIRouter
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


async def authenticate_user(
    db: AsyncSession, email: str, password: str
) -> User | None:
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if user and verify_password(password, user.hashed_password):
        return user
    return None
```

### Token Refresh

```python
@router.post("/refresh")
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db),
):
    payload = verify_token(refresh_token, expected_type="refresh")
    user_id = payload.get("sub")
    user = await db.get(User, int(user_id))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    new_access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": new_access_token, "token_type": "bearer"}
```

## Protecting Routes

```python
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/users", tags=["users"])


# Require authentication
@router.get("/me")
async def read_current_user(
    current_user: User = Depends(get_current_user),
):
    return current_user


# Protect all routes in a router
protected_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_user)],
)
```

## Role-Based Access Control (RBAC)

### Role Model

```python
from enum import StrEnum


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    role: Mapped[str] = mapped_column(String(20), default=Role.USER)
    # ...
```

### Role Checker Dependency

```python
from fastapi import Depends, HTTPException, status


class RoleChecker:
    def __init__(self, allowed_roles: list[Role]):
        self.allowed_roles = allowed_roles

    async def __call__(
        self, user: User = Depends(get_current_user)
    ) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user


# Usage as dependency
allow_admin = RoleChecker([Role.ADMIN])
allow_moderator = RoleChecker([Role.ADMIN, Role.MODERATOR])


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(allow_admin),
    db: AsyncSession = Depends(get_db),
):
    # Only admins can reach this
    ...
```

## API Key Authentication

For service-to-service or simple API key auth:

```python
from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")


async def verify_api_key(
    api_key: str = Security(api_key_header),
) -> str:
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return api_key


@router.get("/external-data", dependencies=[Depends(verify_api_key)])
async def get_external_data():
    ...
```

## Security Best Practices

1. **Use environment variables for secrets** -- never hardcode `secret_key`, API keys, or database credentials.
2. **Set token expiry short for access tokens** (15-30 min) and longer for refresh tokens (7-30 days).
3. **Use HTTPS in production** -- tokens sent over HTTP can be intercepted.
4. **Validate token type** -- prevent refresh tokens from being used as access tokens.
5. **Rate limit auth endpoints** -- prevent brute-force attacks on login.
6. **Hash passwords with bcrypt** -- never use MD5, SHA-256, or other fast hashes for passwords.
7. **Return generic error messages** -- "Incorrect email or password" not "User not found" vs "Wrong password".
8. **Log authentication events** -- track login attempts, failures, and token refreshes.
9. **Invalidate tokens on password change** -- include a token version or `iat` claim.
10. **Use `Annotated` for cleaner dependency injection**:

```python
from typing import Annotated

CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(allow_admin)]


@router.get("/me")
async def read_me(user: CurrentUser):
    return user
```

## Cross-References

- For Pydantic request/response models, consult the `pydantic` skill.
- For database models and sessions, consult the `database` skill.
- For FastAPI routing and middleware, consult the `app-scaffolding` skill.
- For CORS and security middleware, consult the `app-scaffolding` references.
- For testing auth flows, consult the `test-runner` skill.
