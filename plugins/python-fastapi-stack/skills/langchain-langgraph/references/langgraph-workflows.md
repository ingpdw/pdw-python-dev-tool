---
name: langgraph-workflows
description: >
  Reference for advanced LangGraph workflow patterns including multi-agent
  systems, complex state management, conditional routing, persistence,
  error handling, parallel execution, and production deployment.
version: 1.0.0
---

# LangGraph Workflows Reference

## Complex Graph Patterns

### Branching

Route execution to one of several parallel paths based on state. Use `add_conditional_edges` to select branches dynamically:

```python
from langgraph.graph import StateGraph, END

def route_by_intent(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "search":
        return "search_node"
    elif intent == "calculate":
        return "calculator_node"
    elif intent == "generate":
        return "generation_node"
    return END

graph = StateGraph(AgentState)
graph.add_node("classifier", classify_intent)
graph.add_node("search_node", handle_search)
graph.add_node("calculator_node", handle_calculation)
graph.add_node("generation_node", handle_generation)
graph.add_node("formatter", format_response)

graph.set_entry_point("classifier")
graph.add_conditional_edges("classifier", route_by_intent)

# All branches converge on the formatter
graph.add_edge("search_node", "formatter")
graph.add_edge("calculator_node", "formatter")
graph.add_edge("generation_node", "formatter")
graph.add_edge("formatter", END)
```

### Merging

Converge multiple paths into a single node. LangGraph executes the merge node once all upstream nodes along the active path have completed. State reducers handle merging values from different branches:

```python
class ResearchState(TypedDict):
    query: str
    findings: Annotated[list[str], operator.add]  # accumulate from branches
    final_report: str

async def web_search(state: ResearchState) -> dict:
    results = await search_web(state["query"])
    return {"findings": [f"Web: {results}"]}

async def db_search(state: ResearchState) -> dict:
    results = await search_database(state["query"])
    return {"findings": [f"DB: {results}"]}

async def synthesize(state: ResearchState) -> dict:
    report = await llm.ainvoke(
        f"Synthesize these findings: {state['findings']}"
    )
    return {"final_report": report.content}
```

### Subgraphs

Encapsulate reusable workflows as compiled subgraphs and embed them as nodes:

```python
# Define a self-contained research subgraph
research_subgraph = build_research_graph().compile()

# Embed it as a node in a parent graph
parent_graph = StateGraph(ParentState)
parent_graph.add_node("research", research_subgraph)
parent_graph.add_node("summarize", summarize_node)
parent_graph.set_entry_point("research")
parent_graph.add_edge("research", "summarize")
parent_graph.add_edge("summarize", END)
```

Subgraph state is mapped to and from the parent state automatically when the TypedDict keys match. For non-matching keys, wrap the subgraph call in a node function that performs the mapping explicitly.

## Multi-Agent Systems

### Supervisor Pattern

A supervisor agent routes tasks to specialized worker agents based on the task type. The supervisor decides which worker to invoke next and when to finish:

```python
from typing import Literal

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    final_answer: str

WORKERS = ["researcher", "coder", "writer"]

async def supervisor_node(state: SupervisorState) -> dict:
    """Decide which worker to invoke next, or finish."""
    system_prompt = (
        "Decide which agent should act next based on the conversation. "
        f"Choose from: {WORKERS} or 'FINISH' if the task is complete."
    )

    class RouteDecision(BaseModel):
        next: Literal[*WORKERS, "FINISH"]
        reasoning: str

    structured_llm = llm.with_structured_output(RouteDecision)
    decision = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ])
    return {"next_agent": decision.next}

def route_supervisor(state: SupervisorState) -> str:
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]

# Build the supervisor graph
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_agent)
graph.add_node("coder", coder_agent)
graph.add_node("writer", writer_agent)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_supervisor)

# Workers always report back to the supervisor
for worker in WORKERS:
    graph.add_edge(worker, "supervisor")
```

### Hierarchical Agents

Layer supervisors to manage teams of agents. A top-level supervisor delegates to team leads, each of which supervises specialized workers:

```python
# Team-level subgraphs
research_team = build_research_team().compile()
engineering_team = build_engineering_team().compile()

# Top-level supervisor
top_graph = StateGraph(TopState)
top_graph.add_node("director", director_node)
top_graph.add_node("research_team", research_team)
top_graph.add_node("engineering_team", engineering_team)

top_graph.set_entry_point("director")
top_graph.add_conditional_edges("director", route_to_team)
top_graph.add_edge("research_team", "director")
top_graph.add_edge("engineering_team", "director")
```

This pattern scales to arbitrarily deep hierarchies while keeping each subgraph independently testable.

## State Management

### Complex State Schemas

Model rich state with nested structures, optional fields, and multiple reducers:

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class ToolResult(TypedDict):
    tool_name: str
    output: str
    timestamp: float

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: Annotated[list[ToolResult], operator.add]
    current_plan: list[str]          # overwritten each update
    completed_steps: Annotated[list[str], operator.add]
    metadata: dict                   # overwritten each update
    error_count: Annotated[int, operator.add]
    is_complete: bool
```

### Custom Reducers

Write custom reducer functions for specialized merge logic:

```python
def merge_unique(existing: list[str], update: list[str]) -> list[str]:
    """Merge two lists, keeping only unique values."""
    seen = set(existing)
    result = list(existing)
    for item in update:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

class DeduplicatedState(TypedDict):
    sources: Annotated[list[str], merge_unique]
```

### State Channels

LangGraph state keys function as channels. Each node reads from and writes to specific channels. Design state so that nodes only modify the channels they own, reducing conflicts and making the data flow explicit.

A useful pattern: prefix channel names with the owning node or domain (`research_findings`, `code_output`, `review_comments`).

## Conditional Routing

### Router Nodes

Create dedicated router nodes that inspect state and return a routing key:

```python
def quality_gate(state: AgentState) -> str:
    """Route based on output quality assessment."""
    last_output = state["messages"][-1].content
    if len(last_output) < 50:
        return "retry"
    if "error" in last_output.lower():
        return "error_handler"
    return "output"

graph.add_conditional_edges(
    "evaluator",
    quality_gate,
    {
        "retry": "agent",
        "error_handler": "error_node",
        "output": END,
    },
)
```

The optional third argument to `add_conditional_edges` is a mapping from return values to node names. When omitted, the return value must match a node name exactly.

### Dynamic Edge Selection with Multiple Destinations

```python
def fan_out_router(state: AgentState) -> list[str]:
    """Send to multiple nodes simultaneously."""
    destinations = []
    if state.get("needs_search"):
        destinations.append("search")
    if state.get("needs_calculation"):
        destinations.append("calculator")
    if not destinations:
        destinations.append("respond")
    return destinations

graph.add_conditional_edges("planner", fan_out_router)
```

Returning a list sends execution to multiple nodes in parallel (fan-out).

## Persistence

### Checkpointing Strategies

Choose a checkpointer based on the deployment environment:

| Checkpointer | Use Case | Persistence |
|---|---|---|
| `MemorySaver` | Development, testing | In-process memory only |
| `AsyncSqliteSaver` | Single-server deployment | Local SQLite file |
| `AsyncPostgresSaver` | Production, multi-server | PostgreSQL database |

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Development
checkpointer = MemorySaver()

# Production
async def get_checkpointer():
    return AsyncPostgresSaver.from_conn_string(
        "postgresql+asyncpg://user:pass@host:5432/langgraph"
    )
```

### Thread and Checkpoint Management

Every invocation requires a `thread_id` in the config. Optionally specify a `checkpoint_id` to resume from a specific point (time-travel):

```python
# Normal invocation -- continues from latest checkpoint
config = {"configurable": {"thread_id": "conv-123"}}

# Resume from a specific checkpoint (time-travel)
config = {
    "configurable": {
        "thread_id": "conv-123",
        "checkpoint_id": "checkpoint-abc",
    }
}

# List all checkpoints for a thread
checkpoints = [
    cp async for cp in checkpointer.alist({"configurable": {"thread_id": "conv-123"}})
]
```

### PostgresSaver Setup

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def create_agent():
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        await checkpointer.setup()  # create tables on first run
        return graph.compile(checkpointer=checkpointer)
```

## Memory Patterns

### Conversation Memory

The `add_messages` reducer combined with a checkpointer provides automatic conversation memory. Messages accumulate across invocations sharing the same `thread_id`.

To limit memory growth, trim messages before the LLM call:

```python
from langchain_core.messages import trim_messages

async def agent_node(state: AgentState) -> dict:
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,
        token_counter=llm,
        strategy="last",
        start_on="human",
        include_system=True,
    )
    response = await llm_with_tools.ainvoke(trimmed)
    return {"messages": [response]}
```

### Long-Term Memory with Summary

Summarize older messages to compress conversation history:

```python
async def summarize_memory(state: AgentState) -> dict:
    if len(state["messages"]) < 20:
        return {}

    old_messages = state["messages"][:-10]
    summary_prompt = f"Summarize this conversation:\n{old_messages}"
    summary = await llm.ainvoke(summary_prompt)

    # Replace old messages with a summary message
    from langchain_core.messages import RemoveMessage
    remove_msgs = [RemoveMessage(id=m.id) for m in old_messages]
    return {
        "messages": remove_msgs + [SystemMessage(content=f"Summary: {summary.content}")],
    }
```

### Cross-Thread Memory (Shared Store)

Use LangGraph's `Store` interface to persist knowledge across threads:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

app = graph.compile(checkpointer=checkpointer, store=store)

# Inside a node, access the store via config
async def memory_node(state: AgentState, config: dict, *, store) -> dict:
    user_id = config["configurable"]["user_id"]
    memories = await store.asearch(("user_memory", user_id))
    # Use memories in the prompt...
    await store.aput(("user_memory", user_id), "mem-1", {"fact": "Prefers Python"})
    return {}
```

## Error Handling in Graphs

### Retry Nodes

Wrap unreliable operations with retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def reliable_api_call(query: str) -> str:
    return await external_api.search(query)

async def search_with_retry(state: AgentState) -> dict:
    try:
        result = await reliable_api_call(state["query"])
        return {"results": result, "error_count": 0}
    except Exception as e:
        return {"error": str(e), "error_count": 1}
```

### Fallback Edges

Route to fallback nodes when errors occur:

```python
def error_router(state: AgentState) -> str:
    if state.get("error_count", 0) >= 3:
        return "fallback"
    if state.get("error"):
        return "retry"
    return "next_step"

graph.add_conditional_edges("process", error_router, {
    "fallback": "graceful_degradation",
    "retry": "process",
    "next_step": "output",
})
```

### Global Error Handler

Wrap the entire graph invocation with error handling:

```python
async def safe_invoke(app, inputs, config):
    try:
        return await app.ainvoke(inputs, config)
    except GraphRecursionError:
        return {"error": "Agent exceeded maximum steps"}
    except Exception as e:
        logger.exception("Graph execution failed")
        return {"error": str(e)}
```

Set the recursion limit to prevent infinite loops:

```python
config = {
    "configurable": {"thread_id": "123"},
    "recursion_limit": 25,
}
```

## Parallel Execution in Graphs

When multiple nodes have no dependency on each other, LangGraph executes them in parallel automatically. Create fan-out patterns by adding edges from one node to multiple downstream nodes:

```python
# Fan-out: planner sends to multiple workers
graph.add_edge("planner", "web_researcher")
graph.add_edge("planner", "db_analyst")
graph.add_edge("planner", "code_writer")

# Fan-in: all workers converge on synthesizer
graph.add_edge("web_researcher", "synthesizer")
graph.add_edge("db_analyst", "synthesizer")
graph.add_edge("code_writer", "synthesizer")
```

Use list-type state with `operator.add` reducers to accumulate results from parallel nodes:

```python
class ParallelState(TypedDict):
    query: str
    partial_results: Annotated[list[str], operator.add]
    final_result: str
```

Each parallel node appends to `partial_results`; the synthesizer receives the merged list.

## Dynamic Graph Construction

Build graphs programmatically when the structure depends on configuration:

```python
def build_pipeline(tools: list, use_memory: bool = True) -> CompiledGraph:
    """Construct a graph dynamically based on available tools."""

    class DynamicState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        active_tools: list[str]

    graph = StateGraph(DynamicState)
    graph.add_node("agent", create_agent_node(tools))

    if tools:
        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges("agent", should_use_tools)
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    graph.set_entry_point("agent")

    checkpointer = MemorySaver() if use_memory else None
    return graph.compile(checkpointer=checkpointer)
```

This pattern enables building configurable agents where the graph topology adapts to the available tools, feature flags, or user permissions.

## Graph Visualization and Debugging

### Visualization

Generate a visual representation of the graph for documentation and debugging:

```python
# ASCII representation
print(app.get_graph().draw_ascii())

# Mermaid diagram (paste into GitHub markdown or Mermaid Live)
print(app.get_graph().draw_mermaid())

# PNG image (requires graphviz)
png_bytes = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
```

### Debugging with LangSmith

Set environment variables to enable tracing:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls_...
export LANGCHAIN_PROJECT=my-agent
```

Every node execution, LLM call, and tool invocation is logged with inputs, outputs, latency, and token usage. Use LangSmith to inspect full execution traces, compare runs, and identify bottlenecks.

### Step-Through Debugging

Use streaming to inspect each node's output as it executes:

```python
async for step in app.astream(
    {"messages": [HumanMessage(content="Debug this")]},
    config=config,
    stream_mode="updates",
):
    for node_name, state_update in step.items():
        print(f"--- {node_name} ---")
        print(json.dumps(state_update, indent=2, default=str))
```

## Production Deployment Patterns

### Health Checks and Graceful Shutdown

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize the agent and checkpointer
    app.state.checkpointer = await create_checkpointer()
    app.state.agent = build_agent(app.state.checkpointer)
    yield
    # Shutdown: close connections
    await app.state.checkpointer.close()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Rate Limiting and Concurrency Control

Limit concurrent graph executions to protect downstream services:

```python
import asyncio

execution_semaphore = asyncio.Semaphore(10)

@app.post("/chat")
async def chat(request: ChatRequest):
    async with execution_semaphore:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}},
        )
    return {"response": result["messages"][-1].content}
```

### Configuration Management

Externalize model and agent configuration:

```python
from pydantic_settings import BaseSettings

class AgentSettings(BaseSettings):
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    recursion_limit: int = 25
    checkpoint_backend: str = "postgres"
    checkpoint_url: str = "postgresql+asyncpg://..."

    model_config = SettingsConfigDict(env_prefix="AGENT_")
```

## Real-World Examples

### Customer Support Bot

A support bot that classifies tickets, retrieves knowledge base articles, and escalates to humans when needed:

```python
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    category: str
    sentiment: str
    should_escalate: bool
    kb_articles: list[str]

# Graph: classify -> retrieve_kb -> respond -> evaluate
# Conditional: if should_escalate -> human_handoff
# Conditional: if sentiment == "negative" -> escalate_check
```

### Research Agent

An agent that decomposes a research question, searches multiple sources in parallel, and synthesizes a report:

```python
class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    findings: Annotated[list[dict], operator.add]
    draft_report: str
    final_report: str
    revision_count: int

# Graph: decompose -> [web_search, arxiv_search, wiki_search] -> synthesize -> review
# Conditional: if revision_count < 2 and needs_revision -> synthesize (loop)
# Terminal: review -> END
```

### Code Review Agent

An agent that analyzes code diffs, checks for common issues, and generates review comments:

```python
class ReviewState(TypedDict):
    diff: str
    file_analyses: Annotated[list[dict], operator.add]
    security_issues: Annotated[list[str], operator.add]
    style_issues: Annotated[list[str], operator.add]
    review_summary: str

# Graph: parse_diff -> [security_check, style_check, logic_check] -> summarize
# Each checker runs in parallel on the diff
# Summarizer aggregates all findings into a review
```
