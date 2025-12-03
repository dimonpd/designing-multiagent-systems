"""
Memory Example - LangGraph

Equivalent to: examples/agents/memory.py (PicoAgents)

This example demonstrates how to maintain conversation memory across
multiple interactions using LangGraph's checkpointer system.

In PicoAgents, we use MemoryTool with working memory.
In LangGraph, we use MemorySaver checkpointer with thread_id.

Run: python examples/frameworks/langgraph/agents/memory.py

Prerequisites:
    - pip install langgraph langchain-openai python-dotenv
    - Set OPENAI_API_KEY environment variable
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


def get_llm():
    """Create Azure OpenAI LLM client."""
    return AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        api_version="2024-08-01-preview",
        temperature=0,
    )


def main():
    """Demonstrate conversation memory with checkpointing."""
    print("=== Memory Example (LangGraph) ===\n")

    # Create the LLM
    llm = get_llm()

    # Create a memory checkpointer
    # This persists state across invocations
    memory = MemorySaver()

    # Create agent with memory
    agent = create_react_agent(
        model=llm,
        tools=[],  # No tools needed for this example
        checkpointer=memory,
    )

    # Thread ID identifies a conversation session
    config = {"configurable": {"thread_id": "user-123"}}

    print("Starting a conversation with memory...\n")

    # First message - introduce ourselves
    messages = [("user", "Hi! My name is Alice and I love hiking.")]
    print(f"User: {messages[0][1]}")

    result = agent.invoke({"messages": messages}, config)
    response = result["messages"][-1].content
    print(f"Agent: {response}\n")

    # Second message - ask about something else
    messages = [("user", "What are some good hiking trails?")]
    print(f"User: {messages[0][1]}")

    result = agent.invoke({"messages": messages}, config)
    response = result["messages"][-1].content
    print(f"Agent: {response}\n")

    # Third message - test memory recall
    messages = [("user", "Do you remember my name and hobby?")]
    print(f"User: {messages[0][1]}")

    result = agent.invoke({"messages": messages}, config)
    response = result["messages"][-1].content
    print(f"Agent: {response}\n")

    # Show the conversation history from state
    print("=" * 50)
    print("Conversation History (from checkpointer):")
    print("=" * 50)

    # Get the current state snapshot
    state = agent.get_state(config)
    for i, msg in enumerate(state.values["messages"]):
        role = msg.type if hasattr(msg, "type") else "unknown"
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"{i+1}. [{role}]: {content}")


if __name__ == "__main__":
    main()
