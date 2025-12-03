"""
Parallel Agent Example - Google ADK

This example demonstrates parallel execution of multiple agents.
This is a native ADK pattern that efficiently runs agents concurrently.

In PicoAgents, parallel execution is typically done manually with asyncio.
In Google ADK, ParallelAgent is a first-class citizen.

Run: python examples/frameworks/google-adk/orchestration/parallel.py

Prerequisites:
    - pip install google-adk
    - Set GOOGLE_API_KEY environment variable
"""

import asyncio

from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


# Create multiple research agents that run in parallel
market_researcher = LlmAgent(
    name="market_researcher",
    model="gemini-flash-latest",
    instruction=(
        "You are a market researcher. Analyze the market potential and "
        "competitive landscape for the given topic. Be concise (2-3 sentences)."
    ),
    description="Analyzes market potential",
    output_key="market_analysis",
)

technical_researcher = LlmAgent(
    name="technical_researcher",
    model="gemini-flash-latest",
    instruction=(
        "You are a technical researcher. Analyze the technical feasibility "
        "and challenges for the given topic. Be concise (2-3 sentences)."
    ),
    description="Analyzes technical aspects",
    output_key="technical_analysis",
)

social_researcher = LlmAgent(
    name="social_researcher",
    model="gemini-flash-latest",
    instruction=(
        "You are a social impact researcher. Analyze the societal benefits "
        "and potential concerns for the given topic. Be concise (2-3 sentences)."
    ),
    description="Analyzes social impact",
    output_key="social_analysis",
)

# Create parallel agent - runs all sub-agents concurrently
root_agent = ParallelAgent(
    name="research_team",
    description="Parallel research from multiple perspectives",
    sub_agents=[market_researcher, technical_researcher, social_researcher],
)


async def main():
    """Demonstrate parallel execution of multiple research agents."""
    print("=== Parallel Agent Example (Google ADK) ===\n")

    # Create runner
    runner = InMemoryRunner(
        agent=root_agent,
        app_name="parallel_example",
    )

    # Create session
    session = await runner.session_service.create_session(
        app_name="parallel_example",
        user_id="user1",
    )

    # Run the parallel workflow
    task = "Analyze the potential of autonomous delivery drones"
    print(f"Task: {task}")
    print("Running 3 researchers in parallel...\n")
    print("=" * 50)

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=task)],
    )

    # Track responses by agent
    responses = {}
    current_agent = None

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=content,
    ):
        # Track which agent is responding
        if hasattr(event, "author"):
            current_agent = event.author

        # Collect responses
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text and current_agent:
                    if current_agent not in responses:
                        responses[current_agent] = ""
                    responses[current_agent] += part.text

    # Print all responses
    for agent_name, response in responses.items():
        print(f"\n--- {agent_name} ---")
        print(response)

    print("\n" + "=" * 50)
    print("=== All Agents Completed (ran in parallel) ===")


if __name__ == "__main__":
    asyncio.run(main())
