"""
Sequential Workflow Example - Google ADK

Equivalent to: examples/workflows/sequential.py (PicoAgents)

This example demonstrates a sequential workflow where agents process
a task one after another, passing state forward via output_key.

In PicoAgents, we use Workflow.chain() with FunctionSteps.
In Google ADK, we use SequentialAgent with sub_agents.

Run: python examples/frameworks/google-adk/workflows/sequential.py

Prerequisites:
    - pip install google-adk
    - Set GOOGLE_API_KEY environment variable
"""

import asyncio

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


# Create specialized agents
# Each agent stores its output in session state via output_key
researcher = LlmAgent(
    name="researcher",
    model="gemini-flash-latest",
    instruction=(
        "You are a researcher. Provide 2-3 key facts and data points "
        "about the topic. Be concise and factual."
    ),
    description="Researches topics and provides facts",
    output_key="research_output",  # Stores output in session state
)

writer = LlmAgent(
    name="writer",
    model="gemini-flash-latest",
    instruction=(
        "You are a writer. Based on the research below, write a short, "
        "engaging paragraph (2-3 sentences). Make it accessible.\n\n"
        "Research: {research_output}"  # References previous agent's output
    ),
    description="Turns research into engaging content",
    output_key="written_content",
)

editor = LlmAgent(
    name="editor",
    model="gemini-flash-latest",
    instruction=(
        "You are an editor. Review the content below and make minor "
        "improvements for clarity and flow. Output the final polished version.\n\n"
        "Content: {written_content}"  # References writer's output
    ),
    description="Polishes and finalizes content",
    output_key="final_output",
)

# Create sequential workflow using SequentialAgent
# This is equivalent to PicoAgents' Workflow.chain()
root_agent = SequentialAgent(
    name="content_pipeline",
    description="Research, write, and edit content",
    sub_agents=[researcher, writer, editor],
)


async def main():
    """Run a sequential workflow with researcher -> writer -> editor."""
    print("=== Sequential Workflow Example (Google ADK) ===\n")

    # Create runner
    runner = InMemoryRunner(
        agent=root_agent,
        app_name="sequential_example",
    )

    # Create session
    session = await runner.session_service.create_session(
        app_name="sequential_example",
        user_id="user1",
    )

    # Run the workflow
    task = "Write about the benefits of electric vehicles"
    print(f"Task: {task}\n")
    print("=== Workflow Execution ===\n")

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=task)],
    )

    current_agent = None
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=content,
    ):
        # Track which agent is responding
        if hasattr(event, "author") and event.author != current_agent:
            if current_agent is not None:
                print("\n")
            current_agent = event.author
            print(f"--- {current_agent} ---")

        # Print the response
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)

    print("\n=== Workflow Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
