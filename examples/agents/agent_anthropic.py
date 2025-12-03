#!/usr/bin/env python3
"""
Anthropic Claude Agent Example

Shows how to use Claude models with PicoAgents.
Demonstrates both tool calling and structured outputs with Claude Sonnet 4.5.

Requires: ANTHROPIC_API_KEY environment variable
Run: python examples/agents/agent_anthropic.py
"""

import asyncio
import os
from typing import List
from pydantic import BaseModel

from picoagents import Agent, AssistantMessage
from picoagents.llm import AnthropicChatCompletionClient


# Define structured output format
class TravelRecommendation(BaseModel):
    """Structured travel recommendation."""
    destination: str
    best_months: List[str]
    attractions: List[str]
    estimated_budget: str
    travel_tips: List[str]


def get_weather(location: str) -> str:
    """Get current weather for a given location."""
    return f"The weather in {location} is sunny, 75°F with clear skies"


def get_flight_info(origin: str, destination: str) -> str:
    """Get flight information between two cities."""
    return f"Direct flights from {origin} to {destination} available daily, starting at $450"


async def main():
    """Run examples with Claude."""
    print("=== Claude Agent Examples ===\n")

    # Example 1: Basic tool calling
    print("1. Tool Calling Example:")
    print("-" * 40)

    tool_agent = Agent(
        name="travel_assistant",
        description="A travel planning assistant",
        instructions="You are a helpful travel assistant with access to weather and flight information.",
        model_client=AnthropicChatCompletionClient(
            model="claude-sonnet-4-5",  # Supports all features
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ),
        tools=[get_weather, get_flight_info],
        example_tasks=[
            "What's the weather in San Francisco?",
            "Are there flights from NYC to London?",
        ],
    )

    # Simple tool calling with streaming
    async for event in tool_agent.run_stream(
        "What's the weather in Paris and are there flights from San Francisco?",
        stream_tokens=False
    ):
        print(event)

    print("\n" + "=" * 50 + "\n")

    # Example 2: Structured output
    print("2. Structured Output Example:")
    print("-" * 40)

    structured_agent = Agent(
        name="travel_planner",
        description="A travel recommendation agent",
        instructions="You are a travel expert. Provide detailed recommendations for destinations.",
        model_client=AnthropicChatCompletionClient(
            model="claude-sonnet-4-5",  # Required for structured outputs
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ),
        output_format=TravelRecommendation
    )

    response = await structured_agent.run(
        "Recommend a beach vacation in Southeast Asia"
    )

    # Access structured output from the assistant message
    last_message = response.messages[-1]
    if isinstance(last_message, AssistantMessage) and isinstance(last_message.structured_content, TravelRecommendation):
        rec = last_message.structured_content
        print(f"\nStructured Recommendation:")
        print(f"  Destination: {rec.destination}")
        print(f"  Best Months: {', '.join(rec.best_months[:3])}")
        print(f"  Top Attractions: {', '.join(rec.attractions[:3])}")
        print(f"  Budget: {rec.estimated_budget}")
        print(f"  Key Tip: {rec.travel_tips[0]}")
    else:
        # The content is JSON but not parsed as structured_content
        print(f"\nResponse (JSON format):")
        print(f"  {last_message.content[:200]}...")

    print("\n" + "=" * 50 + "\n")

    # Example 3: Streaming with structured output
    print("3. Streaming Structured Output:")
    print("-" * 40)

    print("Getting recommendation for Japan...")
    event_count = 0
    async for event in structured_agent.run_stream(
        "Recommend a cultural trip to Japan",
        stream_tokens=False  # Structured output comes at the end
    ):
        event_count += 1
        if isinstance(event, AssistantMessage) and isinstance(event.structured_content, TravelRecommendation):
            print(f"\n✓ Received structured recommendation:")
            print(f"  Destination: {event.structured_content.destination}")
            print(f"  Best time: {', '.join(event.structured_content.best_months[:2])}")
        # Show a sample of the streaming events
        if event_count <= 3:
            print(f"  Event {event_count}: {str(event)[:80]}...")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("export ANTHROPIC_API_KEY='your-key-here'")
    else:
        asyncio.run(main())