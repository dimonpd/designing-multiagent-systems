"""
Basic Agent Example - Microsoft Agent Framework

Equivalent to: examples/agents/basic-agent.py (PicoAgents)

This example demonstrates creating a simple agent with tools using
Microsoft's Agent Framework. The agent has weather and calculator tools.

Run: python examples/frameworks/agent-framework/agents/basic_agent.py

Prerequisites:
    - pip install agent-framework[azure]
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    - Either set AZURE_OPENAI_API_KEY or run `az login` for Azure CLI auth
"""

import asyncio
import os

from agent_framework import ai_function
from agent_framework.azure import AzureOpenAIChatClient


# Define tools using @ai_function decorator (same pattern as PicoAgents)
@ai_function
def get_weather(location: str) -> str:
    """Get current weather for a given location."""
    return f"The weather in {location} is sunny, 75°F"


@ai_function
def calculate(expression: str) -> str:
    """Perform basic mathematical calculations."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {e}"


async def main():
    """Run example interactions with the agent."""
    print("=== Basic Agent Example (Agent Framework) ===\n")

    # Create Azure OpenAI client
    # Reads from environment variables:
    #   AZURE_OPENAI_ENDPOINT
    #   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    #   AZURE_OPENAI_API_KEY (or uses Azure CLI auth)
    client = AzureOpenAIChatClient(
        deployment_name=os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"
        ),
    )

    # Create agent with tools
    # In Agent Framework, we use client.create_agent() instead of Agent()
    agent = client.create_agent(
        name="basic_assistant",
        instructions=(
            "You are a helpful assistant with access to weather and "
            "calculation tools. Use them when appropriate."
        ),
        tools=[get_weather, calculate],
    )

    print(f"Agent: {agent.name}")
    print(f"Tools: get_weather, calculate\n")

    # Run the agent with a query
    task = "What's the weather in New York and what is 12 * 15?"
    print(f"Task: {task}\n")

    # Non-streaming response
    result = await agent.run(task)
    print(f"Response: {result.text}")

    # Streaming response example
    print("\n--- Streaming Example ---")
    stream_task = "What's the weather in Tokyo?"
    print(f"Task: {stream_task}\n")

    async for update in agent.run_stream(stream_task):
        if update.text:
            print(update.text, end="", flush=True)
    print()  # Final newline


if __name__ == "__main__":
    asyncio.run(main())
