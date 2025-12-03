"""
Sequential Workflow Example - Microsoft Agent Framework

Equivalent to: examples/workflows/sequential.py (PicoAgents)

This example demonstrates a sequential workflow where agents process
a task one after another, passing conversation context forward.

In PicoAgents, we use Workflow.chain() with FunctionSteps.
In Agent Framework, we use SequentialBuilder with agents.

Run: python examples/frameworks/agent-framework/workflows/sequential.py

Prerequisites:
    - pip install agent-framework[azure]
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    - Either set AZURE_OPENAI_API_KEY or run `az login` for Azure CLI auth
"""

import asyncio
import os

from agent_framework import AgentRunUpdateEvent, SequentialBuilder
from agent_framework.azure import AzureOpenAIChatClient


async def main():
    """Run a sequential workflow with researcher -> writer -> editor."""
    print("=== Sequential Workflow Example (Agent Framework) ===\n")

    # Create Azure client
    client = AzureOpenAIChatClient(
        deployment_name=os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"
        ),
    )

    # Create specialized agents
    researcher = client.create_agent(
        name="researcher",
        instructions=(
            "You are a researcher. Provide 2-3 key facts and data points "
            "about the topic. Be concise and factual."
        ),
    )

    writer = client.create_agent(
        name="writer",
        instructions=(
            "You are a writer. Take the research provided and turn it into "
            "a short, engaging paragraph (2-3 sentences). Make it accessible."
        ),
    )

    editor = client.create_agent(
        name="editor",
        instructions=(
            "You are an editor. Review the content and make minor improvements "
            "for clarity and flow. Output the final polished version."
        ),
    )

    # Build sequential workflow using SequentialBuilder
    # This is equivalent to PicoAgents' Workflow.chain()
    workflow = (
        SequentialBuilder()
        .participants([researcher, writer, editor])
        .build()
    )

    # Run the workflow
    task = "Write about the benefits of electric vehicles"
    print(f"Task: {task}\n")
    print("=== Workflow Execution ===\n")

    current_agent = None
    async for event in workflow.run_stream(task):
        if isinstance(event, AgentRunUpdateEvent):
            # Print agent header when switching
            if current_agent != event.executor_id:
                if current_agent is not None:
                    print("\n")
                print(f"--- {event.executor_id} ---")
                current_agent = event.executor_id

            # Print streamed text
            if event.data and event.data.text:
                print(event.data.text, end="", flush=True)

    print("\n\n=== Workflow Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
