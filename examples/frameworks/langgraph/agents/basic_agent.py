"""
Basic Agent Example - LangChain

Equivalent to: examples/agents/basic-agent.py (PicoAgents)

This example demonstrates a basic agent with tool calling using LangChain's
new create_agent function, which provides middleware support and is the
recommended way to build agents.

In PicoAgents, we use BasicAgent with a list of tools.
In LangChain, we use create_agent with tool definitions.

Run: python examples/frameworks/langgraph/agents/basic_agent.py

Prerequisites:
    - pip install langchain langchain-openai python-dotenv
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

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
    """Demonstrate a basic agent with tools using create_agent."""
    print("=== Basic Agent Example (LangChain) ===\n")

    # Create the LLM
    llm = get_llm()

    # Create the agent with tools using the new create_agent API
    # This is the recommended way to build agents in LangChain v1
    agent = create_agent(
        model=llm,
        tools=[get_weather, calculate],
        system_prompt="You are a helpful assistant with access to weather and calculator tools.",
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
        result = agent.invoke({"messages": [HumanMessage(content=task)]})

        # Get the final response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}\n")


if __name__ == "__main__":
    main()
