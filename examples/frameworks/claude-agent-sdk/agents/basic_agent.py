"""
Basic Agent Example - Claude Agent SDK

Equivalent to: examples/agents/basic-agent.py (PicoAgents)

This example demonstrates creating a simple agent with tools using
Anthropic's Claude Agent SDK. The agent has weather and calculator tools
implemented as an SDK MCP server (in-process custom tools).

Run: python examples/frameworks/claude-agent-sdk/agents/basic_agent.py

Prerequisites:
    - pip install claude-agent-sdk
    - Claude Code CLI installed (https://docs.anthropic.com/en/docs/claude-code)
    - Set ANTHROPIC_API_KEY environment variable
"""

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


# Define tools using @tool decorator (SDK MCP server pattern)
# Signature: @tool(name, description, input_schema)
@tool("get_weather", "Get current weather for a given location", {"location": str})
async def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    """Get current weather for a given location."""
    location = args["location"]
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 75°F",
        "london": "Cloudy, 60°F",
        "tokyo": "Rainy, 65°F",
    }
    result = weather_data.get(
        location.lower(),
        f"Weather data not available for {location}"
    )
    return {
        "content": [{"type": "text", "text": f"Weather in {location}: {result}"}],
    }


@tool("calculate", "Perform basic mathematical calculations", {"expression": str})
async def calculate(args: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a mathematical expression."""
    expression = args["expression"]
    try:
        # Safe evaluation of basic math
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return {
                "content": [
                    {"type": "text", "text": f"{expression} = {result}"}
                ],
            }
        return {
            "content": [{"type": "text", "text": "Invalid expression"}],
            "is_error": True,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "is_error": True,
        }


# Create SDK MCP server with our tools
# This runs in-process (no subprocess overhead)
tools_server = create_sdk_mcp_server(
    name="basic_tools",
    version="1.0.0",
    tools=[get_weather, calculate],
)


def display_message(msg: Any) -> None:
    """Display message content in a clean format."""
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(f"Assistant: {block.text}")
            elif isinstance(block, ToolUseBlock):
                print(f"  [Using tool: {block.name}]")
                if block.input:
                    print(f"  [Input: {block.input}]")
    elif isinstance(msg, ResultMessage):
        print(f"\n[Turns: {msg.num_turns}]")
        if msg.total_cost_usd:
            print(f"[Cost: ${msg.total_cost_usd:.4f}]")


async def main():
    """Run example interactions with the agent."""
    print("=== Basic Agent Example (Claude Agent SDK) ===\n")

    # Configure agent options
    # SDK MCP servers are registered via mcp_servers parameter
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt=(
            "You are a helpful assistant with access to weather and "
            "calculation tools. Use them when appropriate. Be concise."
        ),
        mcp_servers={"tools": tools_server},
        allowed_tools=[
            "mcp__tools__get_weather",
            "mcp__tools__calculate",
        ],
        max_turns=10,
    )

    print("Agent configured with tools: get_weather, calculate")
    print("Model: claude-sonnet-4-5\n")

    # Use ClaudeSDKClient for MCP server interactions
    # Run the agent with a query
    task = "What's the weather in New York and what is 12 * 15?"
    print(f"Task: {task}\n")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)

        async for message in client.receive_response():
            display_message(message)

    # Second query example
    print("\n--- Second Query ---")
    task2 = "What's the weather in Tokyo?"
    print(f"Task: {task2}\n")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task2)

        async for message in client.receive_response():
            display_message(message)


if __name__ == "__main__":
    asyncio.run(main())
