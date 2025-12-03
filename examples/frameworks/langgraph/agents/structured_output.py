"""
Structured Output Example - LangGraph

Equivalent to: examples/agents/structured-output.py (PicoAgents)

This example demonstrates how to get structured (typed) responses from
the model using Pydantic models with LangChain's with_structured_output.

Run: python examples/frameworks/langgraph/agents/structured_output.py

Prerequisites:
    - pip install langgraph langchain-openai python-dotenv
    - Set OPENAI_API_KEY environment variable
"""

from pathlib import Path
from typing import List

from dotenv import load_dotenv
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
    print("=== Structured Output Example (LangGraph) ===\n")

    # Create the LLM with structured output
    llm = get_llm()

    # Bind the structured output schema to the model
    structured_llm = llm.with_structured_output(PersonInfo)

    prompt = (
        "Create a profile for a software engineer named Alice who is "
        "28 years old and skilled in Python, JavaScript, and machine learning."
    )

    print(f"Prompt: {prompt}\n")

    # Get structured response
    person: PersonInfo = structured_llm.invoke(prompt)

    print("Structured Output:")
    print(f"  Name: {person.name}")
    print(f"  Age: {person.age}")
    print(f"  Occupation: {person.occupation}")
    print(f"  Skills: {', '.join(person.skills)}")

    print("\n" + "=" * 50)

    # Demonstrate with a different prompt
    prompt2 = (
        "Create a profile for a data scientist named Bob who is "
        "35 years old and specializes in statistics, R, and deep learning."
    )

    print(f"\nPrompt: {prompt2}\n")

    person2: PersonInfo = structured_llm.invoke(prompt2)

    print("Structured Output:")
    print(f"  Name: {person2.name}")
    print(f"  Age: {person2.age}")
    print(f"  Occupation: {person2.occupation}")
    print(f"  Skills: {', '.join(person2.skills)}")


if __name__ == "__main__":
    main()
