"""
Structured Output Example - Google ADK

Equivalent to: examples/agents/structured-output.py (PicoAgents)

This example demonstrates how to get structured (typed) responses from
the model using Pydantic models in Google ADK.

Run: python examples/frameworks/google-adk/agents/structured_output.py

Prerequisites:
    - pip install google-adk
    - Set GOOGLE_API_KEY environment variable
"""

import asyncio
from typing import List

from pydantic import BaseModel, Field

from google import genai
from google.genai import types


class PersonInfo(BaseModel):
    """Structured output model for person information."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")
    skills: List[str] = Field(description="List of the person's key skills")


async def main():
    """Demonstrate structured output with Google Gemini."""
    print("=== Structured Output Example (Google ADK) ===\n")

    # Create Gemini client directly for structured output
    client = genai.Client()

    prompt = (
        "Create a profile for a software engineer named Alice who is "
        "28 years old and skilled in Python, JavaScript, and machine learning."
    )

    print(f"Prompt: {prompt}\n")

    # Use generate_content with response_schema for structured output
    response = await client.aio.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PersonInfo,
        ),
    )

    # Parse the structured response
    if response.text:
        import json

        data = json.loads(response.text)
        person = PersonInfo(**data)

        print("Structured Output:")
        print(f"  Name: {person.name}")
        print(f"  Age: {person.age}")
        print(f"  Occupation: {person.occupation}")
        print(f"  Skills: {', '.join(person.skills)}")
    else:
        print("No response received")


if __name__ == "__main__":
    asyncio.run(main())
