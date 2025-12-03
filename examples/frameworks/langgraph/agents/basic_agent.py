"""
Basic Agent Example - LangGraph

Equivalent to: examples/agents/basic-agent.py (PicoAgents)

This example demonstrates a basic agent with tool calling using LangGraph's
create_react_agent helper, which provides a ReAct-style agent.

In PicoAgents, we use BasicAgent with a list of tools.
In LangGraph, we use create_react_agent with tool definitions.

Run: python examples/frameworks/langgraph/agents/basic_agent.py

Prerequisites:
    - pip install langgraph langchain-openai python-dotenv
    - Set OPENAI_API_KEY environment variable
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
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


# Define tools using LangChain's @tool decorator
@tool
def get_weather(location: str) -> str:
    """Get current weather for a given location."""
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 75°F",
        "london": "Cloudy, 60°F",
        "tokyo": "Rainy, 65°F",
    }
    return weather_data.get(
        location.lower(), f"Weather data not available for {location}"
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safe evaluation of basic math
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Error: {e}"


def main():
    """Demonstrate a basic ReAct agent with tools."""
    print("=== Basic Agent Example (LangGraph) ===\n")

    # Create the LLM
    llm = get_llm()

    # Create the ReAct agent with tools
    agent = create_react_agent(
        model=llm,
        tools=[get_weather, calculate],
    )

    print("Agent created with tools: get_weather, calculate\n")

    # Run the agent with different tasks
    tasks = [
        "What's the weather like in Tokyo?",
        "Calculate 15 * 7 + 23",
        "What's the weather in London and what is 100 / 4?",
    ]

    for task in tasks:
        print(f"Task: {task}")
        print("-" * 40)

        # Invoke the agent
        result = agent.invoke({"messages": [("user", task)]})

        # Get the final response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}\n")


if __name__ == "__main__":
    main()
