"""
Test that from_scratch implementations have compatible APIs with picoagents.

Run: python test_compatibility.py
"""

import asyncio
import sys

# Toggle which implementation to test
USE_FULL_LIBRARY = "--full" in sys.argv

if USE_FULL_LIBRARY:
    print("Testing: picoagents (full library)")
    from picoagents import Agent
    from picoagents.memory import ListMemory
else:
    print("Testing: from_scratch (v4)")
    from ch04_v4_streaming import Agent, ListMemory


def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"


async def test_basic_agent():
    """Test basic agent creation and run."""
    agent = Agent(
        name="test_agent",
        instructions="You are helpful.",
        model="gpt-4.1-mini"
    )
    response = await agent.run("Say hello")
    assert response.final_content, "Should have response content"
    print(f"  ✓ Basic agent: {response.final_content[:50]}...")


async def test_agent_with_tools():
    """Test agent with tools."""
    agent = Agent(
        name="tool_agent",
        instructions="Use tools when asked about weather.",
        model="gpt-4.1-mini",
        tools=[get_weather]
    )
    response = await agent.run("What's the weather in Paris?")
    assert "Paris" in response.final_content or "Sunny" in response.final_content
    print(f"  ✓ Agent with tools: {response.final_content[:50]}...")


async def test_agent_with_memory():
    """Test agent with memory."""
    memory = ListMemory()
    agent = Agent(
        name="memory_agent",
        instructions="Remember what the user tells you.",
        model="gpt-4.1-mini",
        memory=memory
    )

    await agent.run("My name is Bob")
    response = await agent.run("What's my name?")
    assert "Bob" in response.final_content
    print(f"  ✓ Agent with memory: {response.final_content[:50]}...")


async def test_streaming():
    """Test streaming interface."""
    agent = Agent(
        name="stream_agent",
        instructions="Be brief.",
        model="gpt-4.1-mini"
    )

    events = []
    async for item in agent.run_stream("Say hi"):
        events.append(item)

    assert len(events) > 0, "Should yield events"
    print(f"  ✓ Streaming: {len(events)} events yielded")


async def main():
    print("\n--- Running compatibility tests ---\n")

    await test_basic_agent()
    await test_agent_with_tools()
    await test_agent_with_memory()
    await test_streaming()

    print("\n--- All tests passed ---\n")


if __name__ == "__main__":
    asyncio.run(main())
