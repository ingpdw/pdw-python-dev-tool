# Advanced Concurrency Patterns

Detailed reference for advanced asyncio concurrency patterns. For foundational async/await usage, see the parent [SKILL.md](../SKILL.md).

---

## Advanced TaskGroup Patterns

### Nested TaskGroups

Nest groups to create hierarchies where inner-group failures do not automatically cancel outer-group tasks:

```python
import asyncio

async def phase_one() -> list[str]:
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(fetch("url1"))
        t2 = tg.create_task(fetch("url2"))
    return [t1.result(), t2.result()]

async def phase_two(data: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(transform(d)) for d in data]
    return [t.result() for t in tasks]

async def pipeline() -> None:
    async with asyncio.TaskGroup() as tg:
        p1 = tg.create_task(phase_one())
    results = await phase_two(p1.result())
```

### Error Propagation and Partial Results

When a `TaskGroup` raises `ExceptionGroup`, completed tasks still hold their results. Extract partial results from the group:

```python
tasks: list[asyncio.Task] = []

try:
    async with asyncio.TaskGroup() as tg:
        for url in urls:
            tasks.append(tg.create_task(fetch(url)))
except* Exception as eg:
    for exc in eg.exceptions:
        logger.error("Task failed: %s", exc)

# Collect whatever succeeded
results = []
for task in tasks:
    if task.done() and not task.cancelled() and task.exception() is None:
        results.append(task.result())
```

### Dynamic Task Creation Inside TaskGroup

Unlike `gather()`, a `TaskGroup` allows spawning new tasks based on intermediate results:

```python
async def crawl(start_url: str, max_depth: int = 3) -> set[str]:
    visited: set[str] = set()

    async def visit(url: str, depth: int) -> None:
        if depth > max_depth or url in visited:
            return
        visited.add(url)
        links = await fetch_links(url)
        for link in links:
            tg.create_task(visit(link, depth + 1))

    async with asyncio.TaskGroup() as tg:
        tg.create_task(visit(start_url, 0))

    return visited
```

---

## Producer-Consumer with asyncio.Queue

`asyncio.Queue` is the standard mechanism for decoupling producers from consumers in async code.

### Basic Pattern

```python
import asyncio

async def producer(queue: asyncio.Queue[str], items: list[str]) -> None:
    for item in items:
        await queue.put(item)
    await queue.put(None)  # sentinel

async def consumer(queue: asyncio.Queue[str]) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        await process(item)
        queue.task_done()

async def main() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    items = ["a", "b", "c", "d"]

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(queue, items))
        tg.create_task(consumer(queue))
```

### Multiple Producers and Consumers

Scale by running N producers and M consumers. Use one sentinel per consumer to signal shutdown:

```python
NUM_CONSUMERS = 4

async def main() -> None:
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

    async with asyncio.TaskGroup() as tg:
        # Producers
        for source in data_sources:
            tg.create_task(producer(queue, source))

        # Consumers
        consumers = [tg.create_task(consumer(queue)) for _ in range(NUM_CONSUMERS)]

        # Wait for all producers to finish, then send sentinels
        tg.create_task(shutdown_after_producers(queue, NUM_CONSUMERS))

async def shutdown_after_producers(queue: asyncio.Queue, num_consumers: int) -> None:
    await producers_done_event.wait()
    for _ in range(num_consumers):
        await queue.put(None)
```

### Priority Queue

Use `asyncio.PriorityQueue` when tasks have different urgency levels:

```python
import dataclasses

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    payload: str = dataclasses.field(compare=False)

queue: asyncio.PriorityQueue[PrioritizedItem] = asyncio.PriorityQueue()
await queue.put(PrioritizedItem(priority=2, payload="low"))
await queue.put(PrioritizedItem(priority=0, payload="high"))

item = await queue.get()  # PrioritizedItem(priority=0, payload="high")
```

---

## Async Semaphore Rate Limiting

### Basic Rate Limiting

Limit the number of concurrent requests to an external API:

```python
import asyncio
import httpx

MAX_CONCURRENT = 20
sem = asyncio.Semaphore(MAX_CONCURRENT)

async def fetch(client: httpx.AsyncClient, url: str) -> httpx.Response:
    async with sem:
        return await client.get(url)

async def main() -> None:
    async with httpx.AsyncClient() as client:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch(client, url)) for url in urls]
    results = [t.result() for t in tasks]
```

### Token Bucket Rate Limiter

Implement a token bucket for more precise rate control (e.g., 100 requests per second):

```python
class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_event_loop().time()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            await asyncio.sleep(1 / self.rate)

bucket = TokenBucket(rate=100, capacity=100)

async def rate_limited_call(payload: dict) -> dict:
    await bucket.acquire()
    return await api_call(payload)
```

### Sliding Window Rate Limiter

Track request timestamps to enforce a sliding window:

```python
import time as _time
from collections import deque

class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = _time.monotonic()
                while self.timestamps and now - self.timestamps[0] > self.window:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return
            await asyncio.sleep(0.05)
```

---

## Fan-Out / Fan-In Patterns

Distribute work across many coroutines (fan-out), then aggregate results (fan-in).

```python
async def fan_out_fan_in(items: list[str], concurrency: int = 50) -> list[Result]:
    sem = asyncio.Semaphore(concurrency)
    results: list[Result] = []
    lock = asyncio.Lock()

    async def worker(item: str) -> None:
        async with sem:
            result = await process(item)
        async with lock:
            results.append(result)

    async with asyncio.TaskGroup() as tg:
        for item in items:
            tg.create_task(worker(item))

    return results
```

### Chunked Fan-Out

Process items in batches when the total count is very large:

```python
async def chunked_fan_out(
    items: list[str],
    chunk_size: int = 100,
    concurrency: int = 20,
) -> list[Result]:
    all_results: list[Result] = []

    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        sem = asyncio.Semaphore(concurrency)

        async def worker(item: str) -> Result:
            async with sem:
                return await process(item)

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(worker(item)) for item in chunk]

        all_results.extend(t.result() for t in tasks)

    return all_results
```

---

## Structured Concurrency Principles

Structured concurrency guarantees that concurrent operations form a tree: no task outlives the scope that created it.

**Core rules:**

1. Every task has a clear owner (the `TaskGroup` or scope that created it).
2. When a scope exits, all tasks within it are complete or cancelled.
3. Errors propagate upward -- a child failure is visible to the parent.
4. No orphaned tasks -- every task is joined before the program moves on.

```python
# Structured -- tasks are bounded by the async with block
async with asyncio.TaskGroup() as tg:
    tg.create_task(operation_a())
    tg.create_task(operation_b())
# Here, both tasks are guaranteed to be finished.

# Unstructured -- task can outlive the calling coroutine (avoid this)
task = asyncio.create_task(operation_a())
# If the caller returns without awaiting `task`, the task is orphaned.
```

Prefer `TaskGroup` over bare `create_task()` calls. When `create_task()` is necessary (e.g., long-lived background work), track the task in a set and ensure it is cancelled on shutdown.

---

## asyncio.wait() Patterns

`asyncio.wait()` provides fine-grained control over how to wait for a set of tasks.

### FIRST_COMPLETED -- React As Results Arrive

```python
pending: set[asyncio.Task] = {
    asyncio.create_task(fetch(url)) for url in urls
}

while pending:
    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        if task.exception():
            logger.error("Failed: %s", task.exception())
        else:
            handle_result(task.result())
```

### FIRST_EXCEPTION -- Fail Fast

```python
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

for task in done:
    if task.exception():
        # Cancel remaining work
        for p in pending:
            p.cancel()
        raise task.exception()
```

### ALL_COMPLETED with Timeout

```python
done, pending = await asyncio.wait(tasks, timeout=30.0)

for task in pending:
    task.cancel()
    logger.warning("Task timed out: %s", task.get_name())

results = [task.result() for task in done if not task.exception()]
```

---

## Background Task Patterns

### Long-Running Async Tasks

```python
class BackgroundWorker:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while not self._shutdown.is_set():
            try:
                await self._poll_and_process()
            except Exception:
                logger.exception("Worker iteration failed")
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass  # loop again

    async def stop(self) -> None:
        self._shutdown.set()
        if self._task:
            await self._task
```

### Graceful Shutdown

Handle `SIGTERM` and `SIGINT` to shut down cleanly:

```python
import signal

async def main() -> None:
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    worker = BackgroundWorker()
    await worker.start()

    await shutdown_event.wait()
    logger.info("Shutting down...")

    await worker.stop()
    logger.info("Shutdown complete")
```

### Task Registry

Maintain a registry of fire-and-forget tasks to prevent garbage collection and enable clean shutdown:

```python
class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def create_task(self, coro, *, name: str | None = None) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self, timeout: float = 30.0) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
```

---

## Async Database Patterns

### Connection Pool Management

```python
from contextlib import asynccontextmanager
import asyncpg

class Database:
    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
        )

    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()

    @asynccontextmanager
    async def acquire(self):
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self):
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn
```

### SQLAlchemy Async Session

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

engine = create_async_engine("postgresql+asyncpg://user:pass@host/db", pool_size=20)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

@asynccontextmanager
async def get_session():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def get_user(user_id: int) -> User | None:
    async with get_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
```

---

## Async HTTP Client Patterns

### httpx.AsyncClient with Connection Pooling

```python
import httpx

# Reuse the client across requests for connection pooling
async def create_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
        http2=True,
    )
```

### Retry Logic with Exponential Backoff

```python
import random

async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504}),
) -> httpx.Response:
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url)
            if response.status_code not in retry_statuses:
                return response

            last_exc = httpx.HTTPStatusError(
                f"Status {response.status_code}",
                request=response.request,
                response=response,
            )
        except httpx.TransportError as exc:
            last_exc = exc

        if attempt < max_retries:
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            await asyncio.sleep(delay + jitter)

    raise last_exc  # type: ignore[misc]
```

### Concurrent Batch Requests

```python
async def batch_fetch(
    urls: list[str],
    concurrency: int = 20,
) -> dict[str, httpx.Response | Exception]:
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, httpx.Response | Exception] = {}

    async def _fetch(client: httpx.AsyncClient, url: str) -> None:
        async with sem:
            try:
                results[url] = await fetch_with_retry(client, url)
            except Exception as exc:
                results[url] = exc

    async with httpx.AsyncClient() as client:
        async with asyncio.TaskGroup() as tg:
            for url in urls:
                tg.create_task(_fetch(client, url))

    return results
```

---

## Debouncing and Throttling

### Debounce -- Delay Execution Until Inactivity

```python
class Debouncer:
    def __init__(self, delay: float):
        self.delay = delay
        self._task: asyncio.Task | None = None

    async def __call__(self, coro_func, *args, **kwargs) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

        async def _delayed():
            await asyncio.sleep(self.delay)
            await coro_func(*args, **kwargs)

        self._task = asyncio.create_task(_delayed())

debounce = Debouncer(delay=0.5)

# Each call resets the timer; only the last call within 0.5s executes
await debounce(save_to_db, data)
```

### Throttle -- Limit Execution Frequency

```python
class Throttle:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_call: float = 0

    async def __call__(self, coro_func, *args, **kwargs):
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_call = asyncio.get_event_loop().time()
        return await coro_func(*args, **kwargs)

throttle = Throttle(min_interval=1.0)

# Ensures at least 1 second between calls
result = await throttle(send_notification, message)
```

---

## Signal Handling in Async Applications

```python
import signal
import asyncio

async def main() -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def handle_signal(sig: signal.Signals) -> None:
        logger.info("Received %s, initiating shutdown", sig.name)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal, sig)

    # Application logic
    server = await start_server()

    await stop.wait()

    # Graceful shutdown sequence
    server.close()
    await server.wait_closed()

    # Cancel outstanding tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
```

**Note:** `loop.add_signal_handler()` is only available on Unix. On Windows, use `signal.signal()` from a separate thread or rely on `KeyboardInterrupt`.

---

## Profiling Async Code

### asyncio Debug Mode

Enable debug mode to detect common issues (slow callbacks, forgotten awaits):

```python
# Via environment variable
# PYTHONASYNCIODEBUG=1 python app.py

# Via code
asyncio.run(main(), debug=True)
```

Debug mode logs warnings for:
- Coroutines that were never awaited
- Callbacks that take longer than 100ms (configurable via `loop.slow_callback_duration`)
- Unclosed resources (transports, event loops)

### Event Loop Monitoring

Track event loop utilization to detect blocking calls:

```python
import time

class EventLoopMonitor:
    def __init__(self, interval: float = 1.0, threshold: float = 0.1):
        self.interval = interval
        self.threshold = threshold
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._monitor())

    async def _monitor(self) -> None:
        while True:
            t0 = time.monotonic()
            await asyncio.sleep(self.interval)
            elapsed = time.monotonic() - t0
            delta = elapsed - self.interval

            if delta > self.threshold:
                logger.warning(
                    "Event loop was blocked for %.3fs (expected %.3fs, got %.3fs)",
                    delta,
                    self.interval,
                    elapsed,
                )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
```

### Per-Task Timing

```python
import functools
import time

def async_timed(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("%s completed in %.3fs", func.__name__, elapsed)
    return wrapper

@async_timed
async def fetch_data(url: str) -> bytes:
    ...
```

---

## Real-World Patterns

### Async Web Scraper

```python
import httpx
from dataclasses import dataclass, field

@dataclass
class ScrapeResult:
    url: str
    status: int
    content_length: int
    error: str | None = None

@dataclass
class Scraper:
    concurrency: int = 10
    timeout: float = 30.0
    max_retries: int = 2
    _sem: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self._sem = asyncio.Semaphore(self.concurrency)

    async def scrape(self, urls: list[str]) -> list[ScrapeResult]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._fetch(client, url)) for url in urls]
        return [t.result() for t in tasks]

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> ScrapeResult:
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.get(url, follow_redirects=True)
                    return ScrapeResult(
                        url=url,
                        status=resp.status_code,
                        content_length=len(resp.content),
                    )
                except httpx.HTTPError as exc:
                    if attempt == self.max_retries:
                        return ScrapeResult(
                            url=url, status=0, content_length=0, error=str(exc)
                        )
                    await asyncio.sleep(2 ** attempt)
        # unreachable but satisfies type checker
        raise RuntimeError("unreachable")
```

### Async File Processing Pipeline

Process files through multiple stages with backpressure via bounded queues:

```python
from pathlib import Path

async def read_stage(
    paths: list[Path],
    out_queue: asyncio.Queue[tuple[Path, bytes] | None],
) -> None:
    for path in paths:
        data = await asyncio.to_thread(path.read_bytes)
        await out_queue.put((path, data))
    await out_queue.put(None)

async def transform_stage(
    in_queue: asyncio.Queue[tuple[Path, bytes] | None],
    out_queue: asyncio.Queue[tuple[Path, str] | None],
) -> None:
    while True:
        item = await in_queue.get()
        if item is None:
            await out_queue.put(None)
            break
        path, data = item
        text = await asyncio.to_thread(data.decode, "utf-8")
        transformed = text.upper()  # placeholder transform
        await out_queue.put((path, transformed))
        in_queue.task_done()

async def write_stage(
    in_queue: asyncio.Queue[tuple[Path, str] | None],
    output_dir: Path,
) -> None:
    while True:
        item = await in_queue.get()
        if item is None:
            break
        path, text = item
        out_path = output_dir / path.name
        await asyncio.to_thread(out_path.write_text, text)
        in_queue.task_done()

async def process_files(paths: list[Path], output_dir: Path) -> None:
    q1: asyncio.Queue[tuple[Path, bytes] | None] = asyncio.Queue(maxsize=50)
    q2: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue(maxsize=50)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(read_stage(paths, q1))
        tg.create_task(transform_stage(q1, q2))
        tg.create_task(write_stage(q2, output_dir))
```

### Async Pub/Sub

In-process publish-subscribe for decoupled components:

```python
from collections import defaultdict
from typing import Any, Callable, Coroutine

Subscriber = Callable[..., Coroutine[Any, Any, None]]

class PubSub:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._registry = TaskRegistry()

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Subscriber) -> None:
        self._subscribers[topic].remove(handler)

    async def publish(self, topic: str, **kwargs: Any) -> None:
        handlers = self._subscribers.get(topic, [])
        if not handlers:
            return

        async with asyncio.TaskGroup() as tg:
            for handler in handlers:
                tg.create_task(handler(**kwargs))

    async def publish_nowait(self, topic: str, **kwargs: Any) -> None:
        """Fire-and-forget publish -- errors are logged, not raised."""
        handlers = self._subscribers.get(topic, [])
        for handler in handlers:
            async def _safe_call(h=handler):
                try:
                    await h(**kwargs)
                except Exception:
                    logger.exception("Pub/sub handler %s failed", h.__name__)

            self._registry.create_task(_safe_call())

    async def shutdown(self) -> None:
        await self._registry.shutdown()

# Usage
bus = PubSub()

async def on_user_created(user_id: int, email: str) -> None:
    await send_welcome_email(email)

bus.subscribe("user.created", on_user_created)
await bus.publish("user.created", user_id=42, email="user@example.com")
```

### Async Context-Managed Resource Pool

Generic pool pattern for reusable async resources (connections, sessions, channels):

```python
class AsyncPool[T]:
    def __init__(
        self,
        factory: Callable[[], Coroutine[Any, Any, T]],
        close: Callable[[T], Coroutine[Any, Any, None]],
        max_size: int = 10,
    ):
        self._factory = factory
        self._close = close
        self._max_size = max_size
        self._pool: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        resource = await self._get()
        try:
            yield resource
        finally:
            await self._pool.put(resource)

    async def _get(self) -> T:
        # Try to get an existing resource
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Create a new one if under the limit
        async with self._lock:
            if self._size < self._max_size:
                self._size += 1
                return await self._factory()

        # Wait for one to become available
        return await self._pool.get()

    async def close_all(self) -> None:
        while not self._pool.empty():
            resource = self._pool.get_nowait()
            await self._close(resource)
        self._size = 0
```
