---
name: tools
description: >
  Reference for LangChain tool creation, configuration, and integration
  patterns including the @tool decorator, StructuredTool, async tools,
  error handling, built-in tools, retriever tools, and ToolNode usage
  in LangGraph agents.
version: 1.0.0
---

# Tools Reference

## @tool Decorator

The `@tool` decorator is the simplest way to create a LangChain tool. It extracts the tool name from the function name and the schema from the type annotations and docstring.

### Basic Usage

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to look up weather for.
    """
    return fetch_weather_api(city)
```

### Configuration Parameters

```python
@tool(
    name="weather_lookup",           # override function name
    description="Look up weather",    # override docstring description
    return_direct=True,              # return tool output directly to user
    response_format="content",       # "content" or "content_and_artifact"
)
def get_weather(city: str) -> str:
    """This docstring is ignored when description is provided."""
    return fetch_weather_api(city)
```

- `name` -- override the tool name (defaults to the function name).
- `description` -- override the description (defaults to the docstring).
- `return_direct` -- when `True`, the tool output is returned to the user without passing back through the LLM. Useful for tools that produce final answers.
- `response_format` -- set to `"content_and_artifact"` to return both a text summary and a raw artifact (e.g., a DataFrame or image bytes).

### Custom args_schema

Provide a Pydantic model to override the auto-generated argument schema:

```python
from pydantic import BaseModel, Field

class WeatherArgs(BaseModel):
    city: str = Field(description="City name")
    units: str = Field(
        default="celsius",
        description="Temperature units",
        enum=["celsius", "fahrenheit"],
    )

@tool(args_schema=WeatherArgs)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return fetch_weather_api(city, units)
```

The Pydantic model provides richer type information, enum constraints, and descriptions that improve how the LLM selects and parameterizes the tool.

## StructuredTool.from_function()

For maximum control, build tools programmatically with `StructuredTool`:

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class DatabaseQueryArgs(BaseModel):
    table: str = Field(description="Table name to query")
    filters: dict = Field(default_factory=dict, description="Column-value filter pairs")
    limit: int = Field(default=10, ge=1, le=1000, description="Max rows to return")
    order_by: str | None = Field(default=None, description="Column to sort by")


def query_database(table: str, filters: dict = None, limit: int = 10, order_by: str = None) -> list[dict]:
    """Execute a filtered query against the database."""
    return db.query(table, filters=filters or {}, limit=limit, order_by=order_by)


db_query_tool = StructuredTool.from_function(
    func=query_database,
    name="query_database",
    description="Query a database table with optional filters, ordering, and limits",
    args_schema=DatabaseQueryArgs,
    return_direct=False,
)
```

`StructuredTool` is preferable when the tool function already exists and cannot be decorated, or when the schema must be generated dynamically.

## Complex Tool Schemas with Pydantic

Model nested and complex argument structures with Pydantic for tools that accept rich inputs:

```python
from pydantic import BaseModel, Field


class DateRange(BaseModel):
    start: str = Field(description="Start date in YYYY-MM-DD format")
    end: str = Field(description="End date in YYYY-MM-DD format")


class SearchFilters(BaseModel):
    categories: list[str] = Field(
        default_factory=list,
        description="Product categories to include",
    )
    price_range: tuple[float, float] | None = Field(
        default=None,
        description="Min and max price as (min, max)",
    )
    date_range: DateRange | None = Field(
        default=None,
        description="Date range for the search",
    )
    in_stock_only: bool = Field(default=True, description="Only return in-stock items")


class ProductSearchArgs(BaseModel):
    query: str = Field(description="Natural language search query")
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description="Optional search filters",
    )
    limit: int = Field(default=20, ge=1, le=100)


@tool(args_schema=ProductSearchArgs)
def search_products(query: str, filters: SearchFilters = None, limit: int = 20) -> list[dict]:
    """Search the product catalog with optional filters."""
    return catalog.search(query, filters=filters, limit=limit)
```

Nested Pydantic models translate to nested JSON Schema objects, giving the LLM a clear structure for complex tool calls.

## Async Tools

Define async tool functions for I/O-bound operations. LangGraph and LCEL chains call the async variant automatically when using `.ainvoke()` or `.astream()`:

```python
import httpx
from langchain_core.tools import tool

@tool
async def fetch_url(url: str) -> str:
    """Fetch the content of a URL.

    Args:
        url: The URL to fetch content from.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        return response.text[:5000]
```

### Async with StructuredTool

Provide both sync and async implementations via `coroutine`:

```python
import asyncio

def search_sync(query: str) -> list[dict]:
    return db.search(query)

async def search_async(query: str) -> list[dict]:
    return await async_db.search(query)

search_tool = StructuredTool.from_function(
    func=search_sync,
    coroutine=search_async,
    name="search",
    description="Search the database",
)
```

When `coroutine` is provided, async invocations use the async function while sync invocations use the sync function.

## Tool Error Handling

### ToolException

Raise `ToolException` to return an error message to the LLM without crashing the agent. The LLM can then decide to retry or try a different approach:

```python
from langchain_core.tools import tool, ToolException

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: The numerator.
        b: The denominator.
    """
    if b == 0:
        raise ToolException("Cannot divide by zero. Provide a non-zero denominator.")
    return a / b
```

### handle_tool_error

Configure global error handling on a tool to catch all exceptions and convert them to messages:

```python
def handle_search_error(error: Exception) -> str:
    """Convert tool errors into LLM-readable messages."""
    if isinstance(error, ConnectionError):
        return "Search service is temporarily unavailable. Try a different approach."
    if isinstance(error, TimeoutError):
        return "Search timed out. Try a more specific query."
    return f"Search failed: {str(error)}"

@tool(handle_tool_error=handle_search_error)
def web_search(query: str) -> str:
    """Search the web for information."""
    return search_api.search(query)
```

`handle_tool_error` accepts:
- `True` -- return the exception message as a string.
- A string -- return that static string for any error.
- A callable -- call it with the exception to produce an error string.

### Error Handling in ToolNode

`ToolNode` catches `ToolException` automatically and converts it to a `ToolMessage` with the error content, allowing the LLM to self-correct:

```python
from langgraph.prebuilt import ToolNode

# ToolNode handles ToolException by default
tool_node = ToolNode(
    tools=[divide, web_search],
    handle_tool_errors=True,  # default behavior
)
```

## Built-In Tools

LangChain provides pre-built tools for common tasks. Install the corresponding packages as needed.

### TavilySearchResults

```python
# pip install langchain-community tavily-python
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
)

results = await search.ainvoke({"query": "LangGraph production deployment"})
```

### WikipediaQueryRun

```python
# pip install langchain-community wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000))
result = wiki.invoke("Python programming language")
```

### PythonREPLTool

```python
# pip install langchain-experimental
from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()
result = python_repl.invoke("print(2 + 2)")
```

Use `PythonREPLTool` with extreme caution in production -- it executes arbitrary code. Always sandbox execution or restrict to trusted inputs.

## Retriever as Tool

Convert a LangChain retriever into a tool so agents can search document stores:

```python
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Build or load a vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("./index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Wrap the retriever as a tool
retriever_tool = create_retriever_tool(
    retriever,
    name="search_knowledge_base",
    description=(
        "Search the internal knowledge base for information about company policies, "
        "procedures, and product documentation. Use this when the user asks about "
        "internal company information."
    ),
)
```

Write a specific, detailed description so the LLM knows when to use the retriever versus other tools. Vague descriptions lead to over- or under-use.

## API Tools

### RequestsToolkit

Interact with REST APIs directly:

```python
# pip install langchain-community
from langchain_community.agent_toolkits import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper

requests_wrapper = TextRequestsWrapper(
    headers={"Authorization": "Bearer token123"}
)

toolkit = RequestsToolkit(
    requests_wrapper=requests_wrapper,
    allow_dangerous_requests=True,  # required -- acknowledge risk
)

tools = toolkit.get_tools()
# Provides: requests_get, requests_post, requests_put, requests_delete, requests_patch
```

### OpenAPIToolkit

Generate tools from an OpenAPI specification:

```python
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.utilities.requests import TextRequestsWrapper

# Load OpenAPI spec
import yaml
with open("openapi.yaml") as f:
    spec = yaml.safe_load(f)

toolkit = OpenAPIToolkit.from_llm(
    llm=llm,
    spec=spec,
    requests_wrapper=TextRequestsWrapper(),
    allow_dangerous_requests=True,
)

tools = toolkit.get_tools()
```

API tools execute real HTTP requests. Always validate inputs, use authentication, and restrict allowed endpoints in production.

## Tool Binding

### model.bind_tools()

Bind tools to a chat model so it can generate tool calls:

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)
```

### tool_choice

Control which tool the model uses:

```python
# Let the model decide (default)
llm.bind_tools(tools)

# Force a specific tool
llm.bind_tools(tools, tool_choice="add")

# Force the model to call any tool (no plain text response)
llm.bind_tools(tools, tool_choice="any")

# Prevent tool usage (just available for reference)
llm.bind_tools(tools, tool_choice="none")
```

Not all providers support all `tool_choice` values. OpenAI and Anthropic support `"auto"` (default), `"none"`, `"any"` (or `"required"`), and specific tool names.

## Parsing Tool Calls

### AIMessage.tool_calls

After invoking a model with bound tools, inspect the `tool_calls` attribute:

```python
response = llm_with_tools.invoke("What is 5 + 3?")

for call in response.tool_calls:
    print(f"Tool: {call['name']}")
    print(f"Args: {call['args']}")
    print(f"ID:   {call['id']}")
```

`tool_calls` is a list of dicts with `name`, `args`, and `id` keys. The `id` is used to match tool results back to the original call via `ToolMessage`.

### Manual Tool Execution

Execute tool calls manually and construct `ToolMessage` responses:

```python
from langchain_core.messages import ToolMessage

tools_by_name = {t.name: t for t in tools}

for call in response.tool_calls:
    tool = tools_by_name[call["name"]]
    result = tool.invoke(call["args"])
    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=call["id"],
        name=call["name"],
    )
    messages.append(tool_message)
```

The `tool_call_id` field is required -- it links the result to the original tool call so the model can associate inputs with outputs.

## ToolNode in LangGraph

`ToolNode` is a prebuilt LangGraph node that automatically executes tool calls from the last AI message and returns `ToolMessage` results:

```python
from langgraph.prebuilt import ToolNode

tools = [get_weather, search_database, calculate]
tool_node = ToolNode(tools)
```

### How ToolNode Works

1. Read the last message from state (`state["messages"][-1]`).
2. Extract `tool_calls` from the `AIMessage`.
3. Execute each tool call (in parallel when possible).
4. Return a list of `ToolMessage` objects.

### Integration in a Graph

```python
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
)
graph.add_edge("tools", "agent")
```

### Custom Tool Node

For advanced scenarios (logging, rate limiting, pre/post-processing), create a custom tool node:

```python
async def custom_tool_node(state: State) -> dict:
    """Execute tool calls with logging and error handling."""
    last_message = state["messages"][-1]
    tool_messages = []

    for call in last_message.tool_calls:
        tool = tools_by_name.get(call["name"])
        if not tool:
            tool_messages.append(ToolMessage(
                content=f"Unknown tool: {call['name']}",
                tool_call_id=call["id"],
                name=call["name"],
            ))
            continue

        try:
            logger.info(f"Calling tool {call['name']} with {call['args']}")
            result = await tool.ainvoke(call["args"])
            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
                name=call["name"],
            ))
        except Exception as e:
            logger.error(f"Tool {call['name']} failed: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=call["id"],
                name=call["name"],
            ))

    return {"messages": tool_messages}
```

## Multi-Tool Agents

### Routing Between Tools

When an agent has many tools, organize them into categories and use routing to manage complexity:

```python
# Group tools by domain
search_tools = [web_search, knowledge_base_search, code_search]
action_tools = [create_ticket, send_email, update_database]
analysis_tools = [run_sql_query, generate_chart, calculate_statistics]

all_tools = search_tools + action_tools + analysis_tools

# The model sees all tools and selects the appropriate one
llm_with_tools = llm.bind_tools(all_tools)
```

### Tool Selection Guidance

Improve tool selection by providing clear, distinct descriptions and system prompt guidance:

```python
system_prompt = """Available tool categories:
- Search tools: Use for finding information (web_search, knowledge_base_search, code_search)
- Action tools: Use for performing operations (create_ticket, send_email, update_database)
- Analysis tools: Use for data analysis (run_sql_query, generate_chart, calculate_statistics)

Always search for information before taking actions. Use analysis tools for data questions."""
```

### Limiting Tool Iterations

Prevent runaway tool loops by setting a recursion limit or adding an iteration counter to the state:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: Annotated[int, operator.add]

def should_continue(state: AgentState) -> str:
    if state["tool_call_count"] >= 10:
        return "max_reached"
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

async def agent_node(state: AgentState) -> dict:
    response = await llm_with_tools.ainvoke(state["messages"])
    count = len(response.tool_calls) if response.tool_calls else 0
    return {"messages": [response], "tool_call_count": count}
```

## Tool Validation and Safety Patterns

### Input Validation

Validate tool inputs beyond what Pydantic provides:

```python
@tool
def query_database(sql: str) -> list[dict]:
    """Run a read-only SQL query against the database.

    Args:
        sql: A SELECT SQL query. Only SELECT statements are allowed.
    """
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ToolException("Only SELECT queries are allowed.")
    if any(keyword in normalized for keyword in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER"]):
        raise ToolException("Mutating operations are not allowed.")
    return db.execute(sql)
```

### Output Sanitization

Truncate or filter tool output to prevent overwhelming the LLM context:

```python
@tool
def search_documents(query: str) -> str:
    """Search internal documents.

    Args:
        query: Search query string.
    """
    results = search_engine.search(query, limit=10)
    formatted = "\n\n".join(
        f"**{r['title']}**\n{r['content'][:500]}" for r in results
    )
    # Truncate total output to stay within context budget
    if len(formatted) > 3000:
        formatted = formatted[:3000] + "\n\n[Results truncated]"
    return formatted
```

### Rate Limiting

Protect external APIs from excessive tool calls:

```python
import asyncio
import time

class RateLimitedTool:
    def __init__(self, tool_func, calls_per_minute: int = 30):
        self.tool_func = tool_func
        self.semaphore = asyncio.Semaphore(calls_per_minute)
        self.interval = 60.0 / calls_per_minute

    async def __call__(self, *args, **kwargs):
        async with self.semaphore:
            result = await self.tool_func(*args, **kwargs)
            await asyncio.sleep(self.interval)
            return result
```

### Tool Permissions

Restrict tool availability based on user roles or context:

```python
def get_tools_for_user(user_role: str) -> list:
    """Return tools available for the given user role."""
    base_tools = [search_knowledge_base, get_weather]

    if user_role in ("admin", "operator"):
        base_tools.extend([update_database, manage_users])

    if user_role == "admin":
        base_tools.extend([delete_records, modify_config])

    return base_tools

# Build agent with role-appropriate tools
tools = get_tools_for_user(current_user.role)
agent = build_agent(tools)
```
