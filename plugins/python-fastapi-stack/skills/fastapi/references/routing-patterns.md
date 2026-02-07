---
name: routing-patterns
description: >
  Advanced FastAPI routing reference covering nested routers, file uploads, WebSocket
  endpoints, Server-Sent Events, streaming responses, API versioning, and pagination.
version: 1.0.0
---

# Routing Patterns Reference

## Nested Routers and Composition

Compose routers hierarchically to mirror resource relationships:

```python
from fastapi import APIRouter

# Top-level domain router
users_router = APIRouter(prefix="/users", tags=["users"])

# Nested sub-router for user-scoped resources
user_posts_router = APIRouter(prefix="/{user_id}/posts", tags=["user-posts"])

@user_posts_router.get("/")
async def list_user_posts(user_id: int):
    ...

@user_posts_router.post("/", status_code=201)
async def create_user_post(user_id: int, post: PostCreate):
    ...

# Attach the nested router
users_router.include_router(user_posts_router)
```

For deeply nested resources, flatten the path in a single router rather than nesting
more than two levels, to keep paths and dependency chains manageable.

### Router Composition with Shared Dependencies

```python
authenticated_router = APIRouter(dependencies=[Depends(get_current_user)])
authenticated_router.include_router(users_router)
authenticated_router.include_router(items_router)

app.include_router(authenticated_router, prefix="/api/v1")
```

This applies the authentication dependency to every route registered on both
`users_router` and `items_router`.

## Path Parameter Validation and Custom Types

### Constrained Path Parameters

Use `Path()` to add validation rules:

```python
from fastapi import Path
from typing import Annotated

@app.get("/items/{item_id}")
async def get_item(
    item_id: Annotated[int, Path(title="Item ID", ge=1, le=10_000)],
):
    ...
```

### Enum Path Parameters

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model": model_name.value}
```

### Path Convertor for File Paths

Capture full file paths including slashes:

```python
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    # file_path can contain slashes: "documents/2024/report.pdf"
    ...
```

## Query Parameter Models

Bundle related query parameters into a Pydantic model (FastAPI 0.115+):

```python
from fastapi import Query
from pydantic import BaseModel, Field

class ItemFilters(BaseModel):
    q: str | None = None
    category: str | None = None
    min_price: float | None = Field(None, ge=0)
    max_price: float | None = Field(None, ge=0)
    in_stock: bool = True

@app.get("/items")
async def list_items(filters: Annotated[ItemFilters, Query()]):
    ...
```

This generates proper OpenAPI documentation for each query parameter while keeping
the function signature clean.

## File Upload

### Single File

```python
from fastapi import UploadFile, File

@app.post("/upload")
async def upload_file(file: UploadFile):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}
```

### Multiple Files

```python
@app.post("/upload-many")
async def upload_files(files: list[UploadFile]):
    return [{"filename": f.filename, "size": f.size} for f in files]
```

### File with Form Data

```python
from fastapi import Form

@app.post("/upload-with-meta")
async def upload_with_metadata(
    file: UploadFile,
    description: Annotated[str, Form()],
    tags: Annotated[list[str], Form()] = [],
):
    ...
```

### Streaming Upload for Large Files

Process large files in chunks to avoid loading them entirely into memory:

```python
import shutil
from pathlib import Path

UPLOAD_DIR = Path("uploads")

@app.post("/upload-large")
async def upload_large_file(file: UploadFile):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            buffer.write(chunk)
    return {"filename": file.filename, "path": str(dest)}
```

## Response Types

### JSONResponse (default)

```python
from fastapi.responses import JSONResponse

@app.get("/custom-json")
async def custom_json():
    return JSONResponse(
        content={"message": "hello"},
        headers={"X-Custom": "value"},
    )
```

### StreamingResponse

Stream data to the client without buffering the full response in memory:

```python
from fastapi.responses import StreamingResponse
import asyncio

async def generate_data():
    for i in range(100):
        yield f"chunk {i}\n"
        await asyncio.sleep(0.1)

@app.get("/stream")
async def stream_data():
    return StreamingResponse(generate_data(), media_type="text/plain")
```

### FileResponse

Serve files from disk efficiently:

```python
from fastapi.responses import FileResponse

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(
        path=f"files/{filename}",
        filename=filename,
        media_type="application/octet-stream",
    )
```

### HTMLResponse

```python
from fastapi.responses import HTMLResponse

@app.get("/page", response_class=HTMLResponse)
async def get_page():
    return """
    <html><body><h1>Hello</h1></body></html>
    """
```

### RedirectResponse

```python
from fastapi.responses import RedirectResponse

@app.get("/old-path")
async def redirect():
    return RedirectResponse(url="/new-path", status_code=301)
```

## WebSocket Endpoints

### Basic WebSocket

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass  # client disconnected
```

### WebSocket Connection Lifecycle

The full lifecycle: accept, communicate, close. Always handle disconnection
gracefully.

```python
@app.websocket("/ws/{room_id}")
async def room_websocket(ws: WebSocket, room_id: str):
    await ws.accept()
    manager.connect(room_id, ws)
    try:
        while True:
            message = await ws.receive_json()
            await manager.broadcast(room_id, message)
    except WebSocketDisconnect:
        manager.disconnect(room_id, ws)
```

### Broadcasting with a Connection Manager

```python
from dataclasses import dataclass, field
from fastapi import WebSocket

@dataclass
class ConnectionManager:
    connections: dict[str, list[WebSocket]] = field(default_factory=dict)

    def connect(self, room: str, ws: WebSocket):
        self.connections.setdefault(room, []).append(ws)

    def disconnect(self, room: str, ws: WebSocket):
        self.connections.get(room, []).remove(ws)

    async def broadcast(self, room: str, message: dict):
        for ws in self.connections.get(room, []):
            await ws.send_json(message)

manager = ConnectionManager()
```

### WebSocket Authentication

Authenticate during the handshake phase via query parameters or headers:

```python
from fastapi import WebSocket, WebSocketException, status

@app.websocket("/ws")
async def authenticated_ws(ws: WebSocket, token: str | None = None):
    if not token or not verify_token(token):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    await ws.accept()
    ...
```

## Server-Sent Events (SSE)

Implement SSE using `StreamingResponse` with the `text/event-stream` content type:

```python
import asyncio
from fastapi.responses import StreamingResponse

async def event_generator():
    while True:
        data = await get_latest_event()
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(1)

@app.get("/events")
async def sse_endpoint():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
```

### SSE with Named Events and IDs

```python
async def event_stream():
    event_id = 0
    while True:
        event_id += 1
        data = await fetch_update()
        yield f"id: {event_id}\nevent: update\ndata: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.5)
```

## API Versioning Strategies

### URL Prefix Versioning (Recommended)

The simplest and most explicit approach:

```python
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.get("/users")
async def list_users_v1():
    ...

@v2_router.get("/users")
async def list_users_v2():
    ...

app.include_router(v1_router)
app.include_router(v2_router)
```

### Header-Based Versioning

Use a dependency to extract the version from a custom header:

```python
from fastapi import Header, HTTPException

async def get_api_version(
    x_api_version: Annotated[str, Header()] = "1",
) -> int:
    if x_api_version not in ("1", "2"):
        raise HTTPException(400, "Unsupported API version")
    return int(x_api_version)

@app.get("/users")
async def list_users(version: Annotated[int, Depends(get_api_version)]):
    if version == 1:
        return await list_users_v1()
    return await list_users_v2()
```

### Subdomain Versioning

Route via a reverse proxy (nginx, Traefik) or use host-based routing:

```python
from fastapi import Request

@app.get("/users")
async def list_users(request: Request):
    host = request.headers.get("host", "")
    if host.startswith("v2."):
        return await list_users_v2()
    return await list_users_v1()
```

URL prefix versioning is the most practical choice for most projects. It is explicit,
easy to test, and works well with OpenAPI documentation.

## Pagination Patterns

### Offset-Based Pagination

Simple and familiar, but degrades on large datasets:

```python
from pydantic import BaseModel, Field

class PaginationParams(BaseModel):
    offset: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)

@app.get("/items")
async def list_items(pagination: Annotated[PaginationParams, Query()], db: DbSession):
    items = await db.execute(
        select(Item).offset(pagination.offset).limit(pagination.limit)
    )
    total = await db.scalar(select(func.count(Item.id)))
    return {
        "items": items.scalars().all(),
        "total": total,
        "offset": pagination.offset,
        "limit": pagination.limit,
    }
```

### Cursor-Based Pagination

Consistent performance regardless of offset depth. Use an encoded cursor pointing to
the last item:

```python
import base64
from datetime import datetime

def encode_cursor(created_at: datetime, item_id: int) -> str:
    return base64.urlsafe_b64encode(f"{created_at.isoformat()}|{item_id}".encode()).decode()

def decode_cursor(cursor: str) -> tuple[datetime, int]:
    decoded = base64.urlsafe_b64decode(cursor).decode()
    ts, id_ = decoded.split("|")
    return datetime.fromisoformat(ts), int(id_)

@app.get("/items")
async def list_items(
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = None,
    db: DbSession = Depends(get_db),
):
    query = select(Item).order_by(Item.created_at.desc(), Item.id.desc())
    if cursor:
        ts, id_ = decode_cursor(cursor)
        query = query.where(
            (Item.created_at < ts) | ((Item.created_at == ts) & (Item.id < id_))
        )
    query = query.limit(limit + 1)
    results = (await db.execute(query)).scalars().all()

    has_next = len(results) > limit
    items = results[:limit]
    next_cursor = encode_cursor(items[-1].created_at, items[-1].id) if has_next else None

    return {"items": items, "next_cursor": next_cursor}
```

## Content Negotiation

Serve different formats based on the `Accept` header:

```python
from fastapi import Request
from fastapi.responses import JSONResponse, Response

@app.get("/data")
async def get_data(request: Request):
    data = await fetch_data()
    accept = request.headers.get("accept", "application/json")

    if "text/csv" in accept:
        csv_content = convert_to_csv(data)
        return Response(content=csv_content, media_type="text/csv")
    if "application/xml" in accept:
        xml_content = convert_to_xml(data)
        return Response(content=xml_content, media_type="application/xml")

    return JSONResponse(content=data)
```
