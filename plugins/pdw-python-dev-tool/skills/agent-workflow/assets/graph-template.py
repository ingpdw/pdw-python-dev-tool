"""
Production-ready LangGraph agent template.

Provides a complete tool-calling agent with:
- Typed state definition with annotated reducers
- Tool definitions with Pydantic schemas
- Agent node (LLM with bound tools)
- Conditional routing (tool loop vs. final response)
- Checkpointer for conversation persistence
- Async invocation function
- FastAPI integration endpoint with streaming

Usage:
    # Standalone
    import asyncio
    asyncio.run(main())

    # With FastAPI
    uvicorn graph_template:fastapi_app --reload
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Graph state shared across all nodes.

    - messages: conversation history, appended via the add_messages reducer.
    - tool_call_count: running total of tool invocations (summed via operator.add).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: Annotated[int, operator.add]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class SearchArgs(BaseModel):
    query: str = Field(description="The search query string")
    limit: int = Field(default=5, ge=1, le=20, description="Max results to return")


@tool(args_schema=SearchArgs)
async def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Search the internal knowledge base for relevant documents.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.
    """
    # Replace with a real retriever or database call.
    return f"Found {limit} results for '{query}': [placeholder results]"


@tool
async def get_current_time() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


TOOLS = [search_knowledge_base, get_current_time]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Act as a helpful assistant. Use the available tools when the user's "
    "question requires looking up information or performing an action. "
    "Always cite tool results in the final answer."
)


def _build_llm():
    """Construct and return the chat model with bound tools."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def agent_node(state: AgentState) -> dict:
    """Invoke the LLM with the current message history and bound tools."""
    llm_with_tools = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    count = len(response.tool_calls) if response.tool_calls else 0
    return {"messages": [response], "tool_call_count": count}


tool_node = ToolNode(TOOLS)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

MAX_TOOL_ITERATIONS = 10


def should_continue(state: AgentState) -> str:
    """Decide whether to call tools, stop due to limits, or finish."""
    last_message = state["messages"][-1]
    if state.get("tool_call_count", 0) >= MAX_TOOL_ITERATIONS:
        return END
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Construct and compile the agent graph.

    Args:
        checkpointer: Optional checkpointer for state persistence.
                      Defaults to MemorySaver if not provided.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Edges
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Async invocation helper
# ---------------------------------------------------------------------------


async def run_agent(
    message: str,
    thread_id: str = "default",
    agent=None,
) -> str:
    """Send a message to the agent and return the final response text.

    Args:
        message: The user message to send.
        thread_id: Conversation thread identifier for persistence.
        agent: A pre-compiled agent graph. Built on the fly if not provided.
    """
    if agent is None:
        agent = build_graph()

    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse


class ChatRequest(BaseModel):
    message: str = Field(description="User message text")
    thread_id: str = Field(default="default", description="Conversation thread ID")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent on startup; clean up on shutdown."""
    app.state.agent = build_graph()
    yield


fastapi_app = FastAPI(title="LangGraph Agent", lifespan=lifespan)


@fastapi_app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Send a message and receive the complete agent response."""
    agent = fastapi_app.state.agent
    config = {"configurable": {"thread_id": request.thread_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    return {"response": result["messages"][-1].content}


@fastapi_app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Send a message and stream individual tokens as server-sent events."""
    agent = fastapi_app.state.agent
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


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


async def main():
    """Run the agent interactively from the command line."""
    agent = build_graph()
    thread_id = "cli-session"

    print("LangGraph Agent (type 'quit' to exit)")
    print("-" * 40)

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = await run_agent(user_input, thread_id=thread_id, agent=agent)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
