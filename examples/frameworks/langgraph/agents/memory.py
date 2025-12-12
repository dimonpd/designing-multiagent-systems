"""
Memory Example - LangChain

Equivalent to: examples/agents/memory.py (PicoAgents)

This example shows how to implement memory patterns in LangChain that are
equivalent to PicoAgents' ListMemory and ChromaDBMemory.

Key difference: PicoAgents automatically injects memories into the agent's
context. In LangChain, you need to either:
1. Use checkpointer for conversation history (automatic)
2. Inject memories into the system prompt manually
3. Use tools for the agent to access memories

Run: python examples/frameworks/langgraph/agents/memory.py

Prerequisites:
    - pip install langchain langchain-openai langgraph python-dotenv
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

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


# =============================================================================
# Demo 1: ListMemory Equivalent - Inject memories into system prompt
# =============================================================================


def list_memory_example():
    """
    Equivalent to PicoAgents ListMemory.

    In PicoAgents, you do:
        memory = ListMemory()
        memory.add(MemoryContent(content="User loves pirate responses"))
        agent = Agent(..., memory=memory)

    In LangChain, the closest equivalent is to inject memories into the
    system prompt, since there's no automatic memory injection.
    """
    print("=== LIST MEMORY EXAMPLE ===")
    print("(Equivalent to PicoAgents ListMemory)\n")

    llm = get_llm()

    # Simulate a list of memories (like PicoAgents ListMemory)
    memories = [
        "User loves pirate-style responses",
        "User prefers short answers",
    ]

    # Inject memories into system prompt (manual equivalent of automatic injection)
    memory_context = "\n".join(f"- {m}" for m in memories)
    system_prompt = f"""You are a helpful assistant.

Use the following context about the user:
{memory_context}"""

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=system_prompt,
    )

    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content="What's 2+2?")]})

    print(f"System prompt includes {len(memories)} memories")
    print(f"Response: {result['messages'][-1].content}")
    print()


# =============================================================================
# Demo 2: Conversation Memory - Using checkpointer
# =============================================================================


def conversation_memory_example():
    """
    Demonstrate conversation memory using checkpointer.

    This maintains conversation history across multiple turns,
    similar to how PicoAgents maintains context within a session.
    """
    print("=== CONVERSATION MEMORY EXAMPLE ===")
    print("(Using MemorySaver checkpointer)\n")

    llm = get_llm()

    # MemorySaver persists conversation across invocations
    checkpointer = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a helpful assistant who remembers the conversation.",
        checkpointer=checkpointer,
    )

    # Thread ID identifies the conversation
    config: RunnableConfig = {"configurable": {"thread_id": "user-123"}}

    # Turn 1
    print("User: My name is Alice and I love hiking.")
    result = agent.invoke(
        {"messages": [HumanMessage(content="My name is Alice and I love hiking.")]},
        config,
    )
    print(f"Agent: {result['messages'][-1].content}\n")

    # Turn 2 - agent should remember from Turn 1
    print("User: What outdoor activities would you suggest for me?")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What outdoor activities would you suggest for me?")]},
        config,
    )
    print(f"Agent: {result['messages'][-1].content}\n")

    # Turn 3 - test recall
    print("User: Do you remember my name?")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Do you remember my name?")]},
        config,
    )
    print(f"Agent: {result['messages'][-1].content}\n")

    # Show conversation history
    state = agent.get_state(config)
    print(f"Conversation has {len(state.values['messages'])} messages total")
    print()


# =============================================================================
# Demo 3: Semantic Memory - Using ChromaDB directly (like PicoAgents)
# =============================================================================


def chromadb_memory_example():
    """
    Equivalent to PicoAgents ChromaDBMemory.

    In PicoAgents:
        memory = ChromaDBMemory(collection_name="demo")
        memory.add(MemoryContent(content="Alice is a software engineer"))
        agent = Agent(..., memory=memory)
        # Agent automatically queries memory and injects relevant context

    In LangChain, you'd use ChromaDB directly and inject results into prompt
    or provide as tool results. This example shows the manual approach.
    """
    print("=== CHROMADB MEMORY EXAMPLE ===")
    print("(Equivalent to PicoAgents ChromaDBMemory)\n")

    # Check if chromadb is available
    try:
        import chromadb
    except ImportError:
        print("ChromaDB not available. Install with: pip install chromadb")
        print("Skipping this example.\n")
        return

    llm = get_llm()

    # Create ChromaDB client and collection
    client = chromadb.Client()
    collection = client.create_collection(name="demo_memory")

    # Add facts (like PicoAgents memory.add())
    facts = [
        "Alice works as a software engineer at TechCorp",
        "Bob is a data scientist who loves Python",
        "Charlie is a product manager focusing on AI products",
        "The office has a coffee machine on the 3rd floor",
    ]

    collection.add(
        documents=facts,
        ids=[f"fact-{i}" for i in range(len(facts))],
    )

    print(f"Added {len(facts)} facts to memory")

    # Query memory (like PicoAgents memory.query())
    query = "programming languages"
    results: Any = collection.query(query_texts=[query], n_results=2)

    print(f"\nQuery: '{query}'")
    print("Found relevant memories:")
    for doc in results["documents"][0]:
        print(f"  - {doc}")

    # Inject results into agent context
    memory_context = "\n".join(f"- {doc}" for doc in results["documents"][0])
    system_prompt = f"""You are a helpful assistant.

Relevant context from memory:
{memory_context}

Use this context to answer questions."""

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=system_prompt,
    )

    # Ask about the memories
    print("\nUser: Who knows about programming?")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Who knows about programming?")]}
    )
    print(f"Agent: {result['messages'][-1].content}")
    print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run memory examples."""
    print("\n" + "=" * 60)
    print("LANGCHAIN - MEMORY EXAMPLES")
    print("=" * 60)
    print("""
Comparison with PicoAgents:

PicoAgents Pattern:
    memory = ListMemory()
    memory.add(MemoryContent(content="fact"))
    agent = Agent(..., memory=memory)  # Auto-injects memories

LangChain Equivalent Patterns:
    1. System Prompt Injection - Manually add memories to prompt
    2. Checkpointer (MemorySaver) - Conversation history across turns
    3. ChromaDB + Manual Injection - Semantic search, then inject

Key Difference: PicoAgents automatically injects relevant memories.
LangChain requires manual injection or custom middleware.
    """)

    import os

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("ERROR: Set AZURE_OPENAI_ENDPOINT environment variable")
        return

    try:
        list_memory_example()
        conversation_memory_example()
        chromadb_memory_example()

        print("=" * 60)
        print("All memory examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
