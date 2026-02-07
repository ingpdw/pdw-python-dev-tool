---
name: async-patterns
description: >
  Provides Python async/await patterns and asyncio best practices. Activated when
  the user asks about async/await patterns, asyncio best practices, concurrent tasks,
  async generators, task groups, async context managers, event loops, running blocking
  code in async, or async testing. Covers asyncio, concurrency, async iterators,
  semaphores, and asynchronous programming patterns in Python.
version: 1.0.0
---

# Python Async/Await Patterns

## Overview

Use asynchronous programming for **I/O-bound** work: network requests, database queries, file operations, and inter-process communication. These operations spend most of their time waiting, and `asyncio` allows other work to proceed during that wait.

Use synchronous (or multiprocessing-based) code for **CPU-bound** work: image processing, cryptographic hashing, data compression, and heavy computation. The Python GIL prevents true parallel execution of Python bytecode in threads, so CPU-bound work does not benefit from `asyncio`.

A useful heuristic: if the bottleneck is *waiting*, go async. If the bottleneck is *computing*, use `multiprocessing` or a task queue.

## Core Concepts

Three building blocks underpin all async Python code:

- **Coroutine** -- a function declared with `async def`. Calling it returns a coroutine object that must be awaited or scheduled.
- **Event loop** -- the scheduler that runs coroutines, handles I/O callbacks, and manages timers. `asyncio.run()` creates and manages the loop.
- **Awaitable** -- anything that can appear after `await`: coroutines, `Task` objects, and `Future` objects.

```python
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(1)  # simulate I/O
    return "result"

async def main() -> None:
    data = await fetch_data()
    print(data)

asyncio.run(main())
```

`asyncio.run()` is the standard entry point. Avoid calling it more than once in a program; instead, structure the application so a single `asyncio.run(main())` drives everything.

## Task Creation

### asyncio.create_task()

Wrap a coroutine in a `Task` to schedule it concurrently on the running event loop:

```python
async def main() -> None:
    task_a = asyncio.create_task(fetch_data("a"))
    task_b = asyncio.create_task(fetch_data("b"))
    result_a = await task_a
    result_b = await task_b
```

Always keep a reference to created tasks. A task without a live reference can be garbage-collected before completion.

### TaskGroup (Python 3.11+)

`TaskGroup` provides structured concurrency -- all tasks are guaranteed to finish (or be cancelled) before the `async with` block exits:

```python
async def main() -> None:
    async with asyncio.TaskGroup() as tg:
        task_a = tg.create_task(fetch_data("a"))
        task_b = tg.create_task(fetch_data("b"))
    # Both tasks are done here
    print(task_a.result(), task_b.result())
```

If any task raises, the remaining tasks are cancelled and the exception propagates as an `ExceptionGroup`.

## Gathering Results

### asyncio.gather()

Run multiple awaitables concurrently and collect results in order:

```python
results = await asyncio.gather(
    fetch_data("a"),
    fetch_data("b"),
    fetch_data("c"),
    return_exceptions=True,  # exceptions returned as values instead of raised
)
```

With `return_exceptions=False` (the default), the first exception cancels the gather. With `return_exceptions=True`, exceptions appear in the results list alongside successful values.

### gather() vs TaskGroup

| Feature | `asyncio.gather()` | `asyncio.TaskGroup` |
|---|---|---|
| Python version | 3.4+ | 3.11+ |
| Error handling | first exception or return_exceptions | ExceptionGroup, cancels siblings |
| Structured concurrency | No | Yes |
| Dynamic task creation | No (fixed at call time) | Yes (create_task inside block) |

Prefer `TaskGroup` in new code targeting Python 3.11+. Use `gather()` when supporting older runtimes or when `return_exceptions=True` behavior is needed.

## Running Blocking Code

Never call blocking functions directly inside a coroutine -- this stalls the entire event loop.

### asyncio.to_thread()

Offload a synchronous function to a separate thread (Python 3.9+):

```python
import time

def cpu_light_blocking() -> str:
    time.sleep(2)  # blocking I/O
    return "done"

async def main() -> None:
    result = await asyncio.to_thread(cpu_light_blocking)
```

### loop.run_in_executor()

For finer control or process-pool execution:

```python
from concurrent.futures import ProcessPoolExecutor

async def main() -> None:
    loop = asyncio.get_running_loop()

    # Thread pool (default)
    result = await loop.run_in_executor(None, blocking_io_func)

    # Process pool for CPU-bound work
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_heavy_func, arg1)
```

## Async Context Managers

### Class-based

```python
class AsyncDBConnection:
    async def __aenter__(self):
        self.conn = await connect_to_db()
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
        return False  # do not suppress exceptions
```

### Decorator-based

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def db_connection(url: str):
    conn = await connect_to_db(url)
    try:
        yield conn
    finally:
        await conn.close()

async def main() -> None:
    async with db_connection("postgres://...") as conn:
        await conn.execute("SELECT 1")
```

## Async Iterators and Generators

### Async generators

```python
async def stream_rows(query: str):
    async with db_connection() as conn:
        cursor = await conn.execute(query)
        async for row in cursor:
            yield row

async def main() -> None:
    async for row in stream_rows("SELECT * FROM users"):
        process(row)
```

### Class-based async iterator

```python
class Countdown:
    def __init__(self, start: int):
        self.current = start

    def __aiter__(self):
        return self

    async def __anext__(self) -> int:
        if self.current <= 0:
            raise StopAsyncIteration
        self.current -= 1
        await asyncio.sleep(0.1)
        return self.current + 1
```

## Concurrency Primitives

### Semaphore -- limit concurrent operations

```python
sem = asyncio.Semaphore(10)

async def rate_limited_fetch(url: str) -> bytes:
    async with sem:
        return await http_get(url)
```

### Lock -- mutual exclusion

```python
lock = asyncio.Lock()

async def update_shared_state():
    async with lock:
        # only one coroutine at a time
        state["counter"] += 1
```

### Event -- notify waiting coroutines

```python
event = asyncio.Event()

async def waiter():
    await event.wait()
    print("event fired")

async def trigger():
    event.set()
```

### Condition -- wait for a predicate

```python
condition = asyncio.Condition()

async def consumer():
    async with condition:
        await condition.wait_for(lambda: len(queue) > 0)
        item = queue.pop()
```

## Timeouts

### asyncio.timeout() (Python 3.11+)

```python
async def main() -> None:
    async with asyncio.timeout(5.0):
        data = await slow_operation()
```

### asyncio.wait_for()

```python
try:
    result = await asyncio.wait_for(slow_operation(), timeout=5.0)
except asyncio.TimeoutError:
    print("operation timed out")
```

`asyncio.timeout()` is preferred in Python 3.11+ because it integrates with structured concurrency and is an async context manager that can wrap multiple awaits.

## Error Handling

### ExceptionGroup (Python 3.11+)

`TaskGroup` raises `ExceptionGroup` when child tasks fail:

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(may_fail_a())
        tg.create_task(may_fail_b())
except* ValueError as eg:
    for exc in eg.exceptions:
        log_error(exc)
except* TypeError as eg:
    for exc in eg.exceptions:
        log_error(exc)
```

Use `except*` to handle subgroups of exceptions selectively.

### Task cancellation

```python
task = asyncio.create_task(long_running())
task.cancel()

try:
    await task
except asyncio.CancelledError:
    print("task was cancelled")
```

Inside a coroutine, catch `CancelledError` only to perform cleanup, then re-raise or let it propagate. Swallowing `CancelledError` silently breaks cancellation semantics.

## Common Pitfalls

1. **Blocking the event loop** -- calling `time.sleep()`, synchronous HTTP libraries, or CPU-heavy code directly in a coroutine. Use `asyncio.to_thread()` or `run_in_executor()`.

2. **Forgotten awaits** -- calling an `async def` function without `await` produces a coroutine object that never executes. Enable Python's `-W default` or `asyncio` debug mode to catch these.

3. **Fire-and-forget tasks without references** -- `asyncio.create_task(coro())` without storing the return value risks silent garbage collection. Always assign to a variable or add to a set:
   ```python
   background_tasks: set[asyncio.Task] = set()

   def schedule(coro):
       task = asyncio.create_task(coro)
       background_tasks.add(task)
       task.add_done_callback(background_tasks.discard)
   ```

4. **Mixing sync and async incorrectly** -- calling `asyncio.run()` from within a running loop raises `RuntimeError`. Use `await` or `create_task()` from async code; use `asyncio.run()` only at the top level.

5. **Assuming thread safety** -- asyncio coroutines run on a single thread, so shared state between coroutines does not need locks *unless* using `to_thread()` or `run_in_executor()`, which introduce true threads.

## Testing Async Code

Use `pytest-asyncio` to test coroutines with pytest:

```python
import pytest

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data()
    assert result == "expected"
```

### Async fixtures

```python
@pytest.fixture
async def db_session():
    session = await create_session()
    yield session
    await session.close()

@pytest.mark.asyncio
async def test_query(db_session):
    rows = await db_session.execute("SELECT 1")
    assert len(rows) == 1
```

Configure `pytest-asyncio` mode in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # all async tests detected automatically
```

With `asyncio_mode = "auto"`, the `@pytest.mark.asyncio` decorator is optional -- any `async def test_*` function is treated as an async test.

## Cross-References

- For FastAPI async endpoints, consult the `fastapi` skill.
- For advanced concurrency patterns including producer-consumer queues, fan-out/fan-in, structured concurrency, and async HTTP client patterns, see [references/concurrency.md](references/concurrency.md).
