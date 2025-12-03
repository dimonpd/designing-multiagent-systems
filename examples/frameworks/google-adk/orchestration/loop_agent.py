"""
Loop Agent (Round-Robin) Example - Google ADK

Equivalent to: examples/orchestration/round-robin.py (PicoAgents)

This example demonstrates a loop-based conversation between a poet and
a critic. The poet writes haikus, and the critic provides feedback until
the haiku is approved.

In PicoAgents, we use RoundRobinOrchestrator.
In Google ADK, we use SequentialAgent in a custom loop pattern.

Run: python examples/frameworks/google-adk/orchestration/loop_agent.py

Prerequisites:
    - pip install google-adk
    - Set GOOGLE_API_KEY environment variable
"""

import asyncio

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


# Create poet agent - writes haikus
poet = LlmAgent(
    name="poet",
    model="gemini-flash-latest",
    instruction=(
        "You are a haiku poet. Write a haiku based on the user's request.\n"
        "If there was previous feedback in the conversation, revise your haiku accordingly.\n"
        "Output only the haiku, formatted with each line on a new line."
    ),
    description="Writes and revises haikus",
    output_key="current_haiku",
)

# Create critic agent - provides feedback
critic = LlmAgent(
    name="critic",
    model="gemini-flash-latest",
    instruction=(
        "You are a haiku critic. Review the haiku that was just written.\n\n"
        "Haiku to review:\n{current_haiku}\n\n"
        "Provide 2-3 specific, actionable suggestions for improvement. "
        "Focus on imagery, syllable count (5-7-5), seasonal words, or emotional impact.\n\n"
        "If the haiku is excellent and needs no changes, respond with exactly: APPROVED\n"
        "Otherwise, provide constructive feedback."
    ),
    description="Critiques haikus and approves when ready",
)

# Create a sequential agent that runs poet then critic
# We'll run this multiple times to simulate round-robin
round_agent = SequentialAgent(
    name="haiku_round",
    description="One round of poet writing and critic reviewing",
    sub_agents=[poet, critic],
)


async def main():
    """Demonstrate iterative poet/critic collaboration."""
    print("=== Loop Agent (Round-Robin) Example (Google ADK) ===\n")

    # Create runner
    runner = InMemoryRunner(
        agent=round_agent,
        app_name="loop_example",
    )

    # Create session
    session = await runner.session_service.create_session(
        app_name="loop_example",
        user_id="user1",
    )

    # Run the workflow
    task = "Write a haiku about cherry blossoms in spring"
    print(f"Task: {task}")
    print("Poet and Critic collaboration:\n")
    print("=" * 50)

    max_iterations = 4
    approved = False

    for iteration in range(max_iterations):
        print(f"\n--- Round {iteration + 1} ---")

        # Build message - first round is the task, subsequent rounds ask for revision
        if iteration == 0:
            message = task
        else:
            message = "Please revise the haiku based on the feedback."

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )

        current_agent = None
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
        ):
            # Track which agent is responding
            if hasattr(event, "author") and event.author != current_agent:
                current_agent = event.author
                print(f"\n[{current_agent}]")

            # Print the response
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(part.text)

                        # Check for approval
                        if "APPROVED" in part.text.upper():
                            approved = True

        if approved:
            print("\n" + "=" * 50)
            print("Haiku approved!")
            break

    if not approved:
        print("\n" + "=" * 50)
        print(f"Max iterations ({max_iterations}) reached.")

    print("=== Loop Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
