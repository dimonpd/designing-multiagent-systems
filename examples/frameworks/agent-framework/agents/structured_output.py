"""
Structured Output Example - Microsoft Agent Framework

Equivalent to: examples/agents/structured-output.py (PicoAgents)

This example demonstrates how to get structured (typed) responses from
the model using Pydantic models in Agent Framework.

Run: python examples/frameworks/agent-framework/agents/structured_output.py

Prerequisites:
    - pip install agent-framework[azure]
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    - Either set AZURE_OPENAI_API_KEY or run `az login` for Azure CLI auth
"""

import asyncio
import os
from typing import List

from pydantic import BaseModel, Field

from agent_framework import ChatOptions
from agent_framework.azure import AzureOpenAIChatClient


class PersonInfo(BaseModel):
    """Structured output model for person information."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")
    skills: List[str] = Field(description="List of the person's key skills")


async def main():
    """Demonstrate structured output with Azure OpenAI."""
    print("=== Structured Output Example (Agent Framework) ===\n")

    # Create Azure client
    client = AzureOpenAIChatClient(
        deployment_name=os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"
        ),
    )

    # Create user message
    prompt = (
        "Create a profile for a software engineer named Alice who is "
        "28 years old and skilled in Python, JavaScript, and machine learning."
    )

    print(f"Prompt: {prompt}\n")

    # Call the model with structured output using ChatOptions
    # In Agent Framework, we use response_format instead of output_format
    result = await client.get_response(
        messages=prompt,
        chat_options=ChatOptions(
            response_format=PersonInfo,
        ),
    )

    # The result.value will be a PersonInfo object
    if result.value and isinstance(result.value, PersonInfo):
        person = result.value
        print("Structured Output:")
        print(f"  Name: {person.name}")
        print(f"  Age: {person.age}")
        print(f"  Occupation: {person.occupation}")
        print(f"  Skills: {', '.join(person.skills)}")
    else:
        print(f"Raw response: {result if result else 'No response'}")


if __name__ == "__main__":
    asyncio.run(main())
