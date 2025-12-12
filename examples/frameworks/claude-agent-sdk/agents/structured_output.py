"""
Structured Output Example - Claude Agent SDK

Equivalent to: examples/agents/structured_output.py (PicoAgents)

This example demonstrates getting structured JSON responses from Claude
using the Claude Agent SDK's output_format option with JSON schema.

Run: python examples/frameworks/claude-agent-sdk/agents/structured_output.py

Prerequisites:
    - pip install claude-agent-sdk
    - Claude Code CLI installed (https://docs.anthropic.com/en/docs/claude-code)
    - Set ANTHROPIC_API_KEY environment variable
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    query,
)

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


# Define JSON schema for weather response
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The location for the weather report",
        },
        "temperature": {
            "type": "number",
            "description": "Temperature in Fahrenheit",
        },
        "conditions": {
            "type": "string",
            "description": "Weather conditions (sunny, cloudy, rainy, etc.)",
        },
        "humidity": {
            "type": "number",
            "description": "Humidity percentage",
        },
        "wind_speed": {
            "type": "number",
            "description": "Wind speed in mph",
        },
    },
    "required": ["location", "temperature", "conditions"],
}

# Define schema for book analysis
BOOK_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "genre": {"type": "string"},
        "themes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
        "rating": {
            "type": "number",
            "minimum": 1,
            "maximum": 5,
        },
    },
    "required": ["title", "author", "genre", "themes", "summary"],
}


async def weather_example():
    """Get structured weather data."""
    print("=== Weather Structured Output ===\n")

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt=(
            "You are a weather service. Provide realistic weather data "
            "in the exact JSON format requested."
        ),
        permission_mode="bypassPermissions",
        max_turns=3,
        output_format={
            "type": "json_schema",
            "schema": WEATHER_SCHEMA,
        },
    )

    prompt = "What's the weather like in San Francisco today?"
    print(f"Query: {prompt}\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            # Structured output is available in ResultMessage
            if message.structured_output:
                weather = message.structured_output
                print("Structured Response:")
                print(json.dumps(weather, indent=2))
                print(f"\nParsed: {weather.get('temperature')}°F, "
                      f"{weather.get('conditions')} in "
                      f"{weather.get('location')}")


async def book_analysis_example():
    """Get structured book analysis."""
    print("\n=== Book Analysis Structured Output ===\n")

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt=(
            "You are a literary analyst. Analyze books and provide "
            "structured analysis in the exact JSON format requested."
        ),
        permission_mode="bypassPermissions",
        max_turns=3,
        output_format={
            "type": "json_schema",
            "schema": BOOK_ANALYSIS_SCHEMA,
        },
    )

    prompt = "Analyze the book '1984' by George Orwell"
    print(f"Query: {prompt}\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.structured_output:
                analysis = message.structured_output
                print("Structured Response:")
                print(json.dumps(analysis, indent=2))

                # Access typed fields
                print(f"\nTitle: {analysis.get('title')}")
                print(f"Author: {analysis.get('author')}")
                print(f"Genre: {analysis.get('genre')}")
                print(f"Themes: {', '.join(analysis.get('themes', []))}")
                if analysis.get('rating'):
                    print(f"Rating: {'⭐' * int(analysis['rating'])}")


async def list_extraction_example():
    """Extract a list of items with structured output."""
    print("\n=== List Extraction Structured Output ===\n")

    list_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["name", "category", "priority"],
                },
            },
            "total_count": {"type": "integer"},
        },
        "required": ["items", "total_count"],
    }

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="Extract and categorize items from the user's text.",
        permission_mode="bypassPermissions",
        max_turns=3,
        output_format={
            "type": "json_schema",
            "schema": list_schema,
        },
    )

    prompt = (
        "I need to: finish the quarterly report (urgent), "
        "buy groceries, schedule dentist appointment, "
        "review the new hire's code (important), "
        "and water the plants."
    )
    print(f"Query: {prompt}\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.structured_output:
                data = message.structured_output
                print("Extracted Items:")
                for item in data.get("items", []):
                    priority_emoji = {
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢"
                    }.get(item.get("priority"), "⚪")
                    print(f"  {priority_emoji} {item.get('name')} "
                          f"[{item.get('category')}]")
                print(f"\nTotal: {data.get('total_count')} items")


async def main():
    """Run all structured output examples."""
    print("=== Structured Output Examples (Claude Agent SDK) ===\n")
    print("Note: Structured outputs require claude-sonnet-4-5 or "
          "claude-opus-4-1\n")

    await weather_example()
    await book_analysis_example()
    await list_extraction_example()


if __name__ == "__main__":
    asyncio.run(main())
