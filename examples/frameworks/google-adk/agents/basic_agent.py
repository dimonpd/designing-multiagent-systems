"""
Basic Agent Example - Google ADK

Equivalent to: examples/agents/basic-agent.py (PicoAgents)

This example demonstrates creating a simple agent with tools using
Google's Agent Development Kit (ADK). The agent has weather and calculator tools.

Run: python examples/frameworks/google-adk/agents/basic_agent.py

Prerequisites:
    - pip install google-adk
    - Set GOOGLE_API_KEY environment variable
"""

import asyncio

from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.adk.tools import ToolContext
from google.genai import types


# Define tools as functions (ADK auto-wraps them)
# Note: ADK tools can optionally receive ToolContext for state access
def get_weather(location: str, tool_context: ToolContext) -> str:
    """Get current weather for a given location."""
    return f"The weather in {location} is sunny, 75°F"


def calculate(expression: str) -> str:
    """Perform basic mathematical calculations."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {e}"


# Create agent at module level
root_agent = Agent(
    name="basic_assistant",
    model="gemini-flash-latest",
    description="A helpful assistant with weather and calculator tools",
    instruction=(
        "You are a helpful assistant with access to weather and "
        "calculation tools. Use them when appropriate."
    ),
    tools=[get_weather, calculate],
)


async def main():
    """Run example interactions with the agent."""
    print("=== Basic Agent Example (Google ADK) ===\n")

    print(f"Agent: {root_agent.name}")
    print(f"Tools: get_weather, calculate\n")

    # Create a runner to execute the agent
    runner = InMemoryRunner(
        agent=root_agent,
        app_name="basic_example",
    )

    # Create a session
    session = await runner.session_service.create_session(
        app_name="basic_example",
        user_id="user1",
    )

    # Run the agent with a query
    task = "What's the weather in New York and what is 12 * 15?"
    print(f"Task: {task}\n")

    # Create user message
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=task)],
    )

    # Run and collect response
    response_text = ""
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_text = part.text

    print(f"Response: {response_text}")

    # Second query in same session
    print("\n--- Second Query ---")
    stream_task = "What's the weather in Tokyo?"
    print(f"Task: {stream_task}\n")

    content2 = types.Content(
        role="user",
        parts=[types.Part.from_text(text=stream_task)],
    )

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=content2,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(f"Response: {part.text}")


if __name__ == "__main__":
    asyncio.run(main())
