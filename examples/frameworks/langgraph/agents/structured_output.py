"""
Structured Output Example - LangChain

Equivalent to: examples/agents/structured-output.py (PicoAgents)

This example demonstrates how to get structured (typed) responses from
the model using Pydantic models with LangChain's create_agent.

The new create_agent API supports response_format parameter for structured output,
which automatically handles tool-based or provider-specific structured output.

Run: python examples/frameworks/langgraph/agents/structured_output.py

Prerequisites:
    - pip install langchain langchain-openai python-dotenv
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables
"""

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

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


class PersonInfo(BaseModel):
    """Structured output model for person information."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")
    skills: List[str] = Field(description="List of the person's key skills")


def main():
    """Demonstrate structured output with Pydantic models."""
    print("=== Structured Output Example (LangChain) ===\n")

    # Create the LLM
    llm = get_llm()

    # Create agent with structured output using response_format
    # This uses the new create_agent API which handles structured output automatically
    agent = create_agent(
        model=llm,
        tools=[],  # No tools needed - structured output is handled separately
        system_prompt="You are a helpful assistant that creates person profiles.",
        response_format=PersonInfo,  # Specify the Pydantic model for structured output
    )

    prompt = (
        "Create a profile for a software engineer named Alice who is "
        "28 years old and skilled in Python, JavaScript, and machine learning."
    )

    print(f"Prompt: {prompt}\n")

    # Get structured response
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})

    # Access the structured response directly from state
    person = result.get("structured_response")
    if person:
        print("Structured Output:")
        print(f"  Name: {person.name}")
        print(f"  Age: {person.age}")
        print(f"  Occupation: {person.occupation}")
        print(f"  Skills: {', '.join(person.skills)}")
    else:
        # Fallback to message content if structured_response not available
        print(f"Response: {result['messages'][-1].content}")

    print("\n" + "=" * 50)

    # Demonstrate with a different prompt
    prompt2 = (
        "Create a profile for a data scientist named Bob who is "
        "35 years old and specializes in statistics, R, and deep learning."
    )

    print(f"\nPrompt: {prompt2}\n")

    result2 = agent.invoke({"messages": [HumanMessage(content=prompt2)]})

    person2 = result2.get("structured_response")
    if person2:
        print("Structured Output:")
        print(f"  Name: {person2.name}")
        print(f"  Age: {person2.age}")
        print(f"  Occupation: {person2.occupation}")
        print(f"  Skills: {', '.join(person2.skills)}")
    else:
        print(f"Response: {result2['messages'][-1].content}")


if __name__ == "__main__":
    main()
