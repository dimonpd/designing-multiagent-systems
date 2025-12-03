"""
Memory Example - Microsoft Agent Framework

Equivalent to: examples/agents/memory.py (PicoAgents)

This example demonstrates how to use ContextProvider for memory injection
in Agent Framework. The ContextProvider pattern is similar to PicoAgents'
memory system but with explicit invoking/invoked hooks.

Run: python examples/frameworks/agent-framework/agents/memory.py

Prerequisites:
    - pip install agent-framework[azure]
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    - Either set AZURE_OPENAI_API_KEY or run `az login` for Azure CLI auth
"""

import asyncio
import os
from collections.abc import MutableSequence, Sequence
from typing import Any

from agent_framework import ChatAgent, ChatMessage, Context, ContextProvider
from agent_framework.azure import AzureOpenAIChatClient


class ListMemory(ContextProvider):
    """Simple list-based memory that injects preferences into agent context.

    This is equivalent to PicoAgents' ListMemory - it stores items and
    injects them as additional instructions before each agent call.
    """

    def __init__(self, max_memories: int = 10):
        self.memories: list[str] = []
        self.max_memories = max_memories

    def add(self, content: str) -> None:
        """Add a memory item."""
        self.memories.append(content)
        # Keep only the most recent memories
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        """Called before each agent invocation - inject memories as context."""
        if not self.memories:
            return Context()

        # Format memories as additional instructions
        memory_text = "\n".join(f"- {m}" for m in self.memories)
        instructions = f"Consider the following user preferences:\n{memory_text}"

        return Context(instructions=instructions)

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Called after each agent invocation - could update memories here."""
        # In this simple example, we don't auto-extract memories
        # A more advanced implementation could analyze responses
        pass


async def list_memory_example():
    """Minimal ListMemory example using ContextProvider."""
    print("=== LIST MEMORY EXAMPLE (Agent Framework) ===\n")

    # Create Azure client
    client = AzureOpenAIChatClient(
        deployment_name=os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"
        ),
    )

    # Create memory and add a preference
    memory = ListMemory(max_memories=10)
    memory.add("User loves pirate-style responses")

    # Create agent with memory as context provider
    agent = ChatAgent(
        chat_client=client,
        name="assistant",
        instructions="You are helpful. Use any relevant context.",
        context_providers=memory,
    )

    # Create a thread to maintain conversation state
    thread = agent.get_new_thread()

    # Run a query - the agent should respond in pirate style
    result = await agent.run("What's 2+2?", thread=thread)
    print(f"Response: {result.text}")
    print(f"Memory items: {len(memory.memories)}")

    # Add another preference and ask again
    memory.add("User prefers concise responses")
    result = await agent.run("What's the meaning of life?", thread=thread)
    print(f"\nResponse: {result.text}")


async def main():
    """Run the memory example."""
    await list_memory_example()


if __name__ == "__main__":
    asyncio.run(main())
