---
name: agent-workflow
description: >
  Guides the agent through building LLM-powered applications with LangChain and
  stateful agent workflows with LangGraph. Triggered when the user asks to
  "create an AI agent", "build a LangChain chain", "create a LangGraph workflow",
  "implement tool calling", "build RAG pipeline", "create a multi-agent system",
  "define agent state", "add human-in-the-loop", "implement streaming", or
  mentions LangChain, LangGraph, chains, agents, tools, retrieval augmented
  generation, state graphs, or LLM orchestration.
version: 1.0.0
---

# LangChain / LangGraph

## Overview

LangChain is the standard framework for building applications powered by large language models. It provides abstractions for chat models, prompt templates, output parsers, tool integration, and retrieval augmented generation. LangChain Expression Language (LCEL) enables declarative composition of these components into chains using a pipe syntax.

LangGraph extends LangChain with a graph-based runtime for building stateful, multi-step agent workflows. It models agent logic as a state graph where nodes are functions, edges define control flow, and state is persisted across turns via checkpointing. LangGraph is the recommended approach for any agent that requires loops, branching, tool-calling cycles, human-in-the-loop intervention, or long-running conversations.

Key characteristics:

- LangChain: model abstraction, prompt management, output parsing, tool definitions, retrieval
- LangGraph: state machines for agents, persistence, streaming, human-in-the-loop, multi-agent orchestration
- LCEL: composable pipe syntax for simple prompt-model-parser chains
- Production-ready: built-in checkpointing, error handling, streaming, and observability

## Chat Models

Chat models are the primary interface to LLMs. Always import from `langchain_<provider>` packages rather than the deprecated `langchain.chat_models`.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Anthropic
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
```

### Message Types

LangChain uses typed message objects for chat interactions:

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

messages = [
    SystemMessage(content="Answer concisely."),
    HumanMessage(content="What is LangGraph?"),
]

response = llm.invoke(messages)  # returns AIMessage
```

`AIMessage` includes `content` (the text response), `tool_calls` (list of tool invocations), and `response_metadata` (token usage, model info).

## Prompt Templates

Use `ChatPromptTemplate` to create reusable, parameterized prompts:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions about {topic}. Be concise."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])

# Invoke with variables
messages = prompt.invoke({
    "topic": "Python",
    "question": "What is asyncio?",
    "chat_history": [],
})
```

`MessagesPlaceholder` injects a list of messages at that position -- essential for conversation history and agent scratchpads. Mark it `optional=True` when the variable may be absent.

## Output Parsers

Parse LLM output into structured formats:

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Simple string output
parser = StrOutputParser()

# Pydantic structured output
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1 to 10", ge=1, le=10)
    summary: str = Field(description="Brief summary")

pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

# JSON output
json_parser = JsonOutputParser(pydantic_object=MovieReview)
```

For structured output with tool-calling models, prefer `model.with_structured_output(MovieReview)` over manual parser injection -- it uses the model's native function-calling capability.

```python
structured_llm = llm.with_structured_output(MovieReview)
result = structured_llm.invoke("Review the movie Inception")
# result is a MovieReview instance
```

For Pydantic structured output, consult the `pydantic` skill.

## LangChain Expression Language (LCEL)

LCEL composes components using the pipe (`|`) operator. Each component in the chain must be a `Runnable`.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following to {language}."),
    ("human", "{text}"),
])

chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()

result = chain.invoke({"language": "French", "text": "Hello, world!"})
```

### Key Runnables

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

# Pass input through unchanged (useful for forwarding data)
RunnablePassthrough()

# Wrap any function as a Runnable
RunnableLambda(lambda x: x["text"].upper())

# Run multiple chains in parallel, merge results
RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
)
```

### Chaining with Context

A common pattern passes retrieved context alongside the original question:

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

LCEL chains support `.invoke()`, `.ainvoke()`, `.stream()`, `.astream()`, `.batch()`, and `.abatch()` out of the box.

## Tool Creation

Tools give LLMs the ability to call external functions. Define tools with clear names and descriptions so the model knows when and how to use them.

### @tool Decorator

```python
from langchain_core.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the product database by query string.

    Args:
        query: The search query to match against product names and descriptions.
        limit: Maximum number of results to return.
    """
    return db.search(query, limit=limit)
```

The decorator extracts the function signature and docstring to build the tool schema automatically.

### StructuredTool with Pydantic Schema

For complex argument validation, provide an explicit `args_schema`:

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="Search query string")
    category: str = Field(description="Product category to filter")
    limit: int = Field(default=10, ge=1, le=100)

search_tool = StructuredTool.from_function(
    func=search_database,
    name="search_database",
    description="Search the product database with filters",
    args_schema=SearchArgs,
)
```

### Binding Tools to Models

```python
tools = [search_database, get_weather, calculate]

# Bind tools to a model
llm_with_tools = llm.bind_tools(tools)

# Force a specific tool
llm_with_tools = llm.bind_tools(tools, tool_choice="search_database")

# Parse tool calls from the response
response = llm_with_tools.invoke("Find laptops under $1000")
for tool_call in response.tool_calls:
    print(tool_call["name"], tool_call["args"])
```

See `references/tools.md` for async tools, error handling, built-in tools, retriever tools, and advanced patterns.

## LangGraph Fundamentals

LangGraph models agent workflows as directed graphs. The three core concepts are **state**, **nodes**, and **edges**.

### State Definition

Define the graph state as a `TypedDict`. Use `Annotated` fields with reducer functions to control how node outputs merge into existing state.

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # append messages
    context: str                                           # overwrite on each update
    iteration_count: Annotated[int, operator.add]          # sum values
```

- `add_messages` -- appends new messages, deduplicates by ID, and handles `RemoveMessage`.
- `operator.add` -- sums numeric values from successive updates.
- No annotation -- the latest value overwrites the previous one.

### Nodes

Nodes are functions (sync or async) that receive the current state and return a partial state update:

```python
from langchain_core.messages import AIMessage

async def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current messages."""
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

async def process_node(state: AgentState) -> dict:
    """Post-process the agent output."""
    last_message = state["messages"][-1]
    return {"context": last_message.content, "iteration_count": 1}
```

Nodes return only the keys being updated. The reducer merges the partial update into the full state.

### Edges and Conditional Routing

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("process", process_node)

# Static edge
graph.add_edge("tools", "agent")

# Conditional edge -- route based on state
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "process"

graph.add_conditional_edges("agent", should_continue)

# Terminal edge
graph.add_edge("process", END)

# Set entry point
graph.set_entry_point("agent")
```

### Graph Compilation

Compile the graph into a runnable application, optionally with a checkpointer for state persistence:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

### Invocation

```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "user-123"}}

result = await app.ainvoke(
    {"messages": [HumanMessage(content="Find laptops under $1000")]},
    config=config,
)
```

The `thread_id` links invocations to a persistent conversation. The checkpointer stores state between calls, enabling multi-turn conversations.

## Basic Agent Pattern

The canonical LangGraph agent follows a tool-calling loop: the LLM decides whether to call tools or respond directly.

```python
from langgraph.prebuilt import ToolNode

tools = [search_database, get_weather]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

async def call_model(state: AgentState) -> dict:
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

agent = graph.compile(checkpointer=MemorySaver())
```

See `assets/graph-template.py` for a complete production-ready template with FastAPI integration.

## Checkpointing and Persistence

Checkpointers save the graph state after every node execution, enabling conversation memory, fault recovery, and time-travel debugging.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# In-memory (development)
checkpointer = MemorySaver()

# SQLite (single-server persistence)
checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL (production -- requires langgraph-checkpoint-postgres)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
```

Always provide a `thread_id` in the config to isolate conversations:

```python
config = {"configurable": {"thread_id": "session-abc123"}}
```

## Human-in-the-Loop

Interrupt graph execution to request human approval or input before proceeding:

```python
# Interrupt before the tool node executes
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],
)

# Run until interrupted
result = await app.ainvoke(
    {"messages": [HumanMessage(content="Delete all records")]},
    config=config,
)

# Inspect pending tool calls, get human approval, then resume
result = await app.ainvoke(None, config=config)
```

Use `interrupt_after` to pause after a node completes, allowing inspection of its output before the graph continues.

## Streaming

LangGraph supports multiple streaming modes for real-time output.

### Stream State Updates

```python
async for event in app.astream(
    {"messages": [HumanMessage(content="Summarize this document")]},
    config=config,
):
    for node_name, output in event.items():
        print(f"Node '{node_name}': {output}")
```

### Stream Individual Tokens

```python
async for event in app.astream_events(
    {"messages": [HumanMessage(content="Write a poem")]},
    config=config,
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        if token:
            print(token, end="", flush=True)
```

`astream_events()` provides fine-grained events for every component in the graph: model tokens, tool starts/ends, retriever results, and custom events.

## Integration with FastAPI

Mount a LangGraph agent as a FastAPI endpoint for production serving:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    return {"response": result["messages"][-1].content}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    async def event_generator():
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

For full FastAPI application structure and patterns, consult the `fastapi` skill. For async patterns in agents, consult the `async-patterns` skill.

## Further Reading

- `references/langgraph-workflows.md` -- complex graph patterns, multi-agent systems, production deployment
- `references/tools.md` -- tool creation, built-in tools, error handling, retriever tools
- `assets/graph-template.py` -- production-ready agent template with FastAPI integration

Cross-references:
- **pydantic** skill -- structured output models, validators, serialization
- **async-patterns** skill -- async/await best practices, concurrency primitives
- **fastapi** skill -- API endpoints, middleware, dependency injection
