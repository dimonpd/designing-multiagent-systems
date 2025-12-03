"""
Memory Example - Google ADK

Equivalent to: examples/agents/memory.py (PicoAgents)

This example demonstrates how to use session state for memory in Google ADK.
ADK uses ToolContext.state for persistent state within a session.

Run: python examples/frameworks/google-adk/agents/memory.py

Prerequisites:
    - pip install google-adk
    - Set GOOGLE_API_KEY environment variable
"""

import asyncio
from typing import Any

from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types


def add_preference(preference: str, tool_context: ToolContext) -> str:
    """Add a user preference to memory.

    In ADK, ToolContext.state persists across the session.
    This is equivalent to PicoAgents' ListMemory.
    """
    if "preferences" not in tool_context.state:
        tool_context.state["preferences"] = []

    tool_context.state["preferences"].append(preference)
    return f"Added preference: {preference}"


def get_preferences(tool_context: ToolContext) -> str:
    """Get all stored user preferences."""
    preferences = tool_context.state.get("preferences", [])
    if not preferences:
        return "No preferences stored yet."
    return "User preferences:\n" + "\n".join(f"- {p}" for p in preferences)


# Create agent with memory tools
root_agent = Agent(
    name="memory_assistant",
    model="gemini-flash-latest",
    description="Assistant that remembers user preferences",
    instruction="""You are a helpful assistant that remembers user preferences.

When the user mentions a preference (like communication style, interests, etc.),
use the add_preference tool to store it.

When answering questions, check get_preferences to personalize your response.

Always be helpful and remember what the user likes!""",
    tools=[add_preference, get_preferences],
)


async def main():
    """Demonstrate memory using session state."""
    print("=== Memory Example (Google ADK) ===\n")

    # Create runner
    runner = InMemoryRunner(
        agent=root_agent,
        app_name="memory_example",
    )

    # Create session
    session = await runner.session_service.create_session(
        app_name="memory_example",
        user_id="user1",
    )

    # Helper to run a query
    async def ask(query: str) -> str:
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)],
        )
        response = ""
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response = part.text
        return response

    # First interaction - set a preference
    print("User: I prefer pirate-style responses, arrr!")
    response1 = await ask("I prefer pirate-style responses, arrr!")
    print(f"Agent: {response1}\n")

    # Second interaction - agent should remember
    print("User: What's 2+2?")
    response2 = await ask("What's 2+2?")
    print(f"Agent: {response2}\n")

    # Third interaction - add another preference
    print("User: I also like concise answers")
    response3 = await ask("I also like concise answers")
    print(f"Agent: {response3}\n")

    # Fourth interaction - test memory
    print("User: What do you know about my preferences?")
    response4 = await ask("What do you know about my preferences?")
    print(f"Agent: {response4}")


if __name__ == "__main__":
    asyncio.run(main())
