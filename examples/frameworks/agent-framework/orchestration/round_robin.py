"""
Round-Robin Orchestration Example - Microsoft Agent Framework

Equivalent to: examples/orchestration/round-robin.py (PicoAgents)

This example demonstrates a round-robin conversation between a poet and
a critic. The poet writes haikus, and the critic provides feedback until
the haiku is approved.

In PicoAgents, we use RoundRobinOrchestrator.
In Agent Framework, we use SequentialBuilder with cyclic workflow for
true round-robin, or WorkflowBuilder for more control.

Run: python examples/frameworks/agent-framework/orchestration/round_robin.py

Prerequisites:
    - pip install agent-framework[azure]
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    - Either set AZURE_OPENAI_API_KEY or run `az login` for Azure CLI auth
"""

import asyncio
import os

from agent_framework import (
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentRunUpdateEvent,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    executor,
)
from agent_framework.azure import AzureOpenAIChatClient


async def main():
    """Demonstrate round-robin conversation between poet and critic."""
    print("=== Round-Robin Orchestration Example (Agent Framework) ===\n")

    # Create Azure client
    client = AzureOpenAIChatClient(
        deployment_name=os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"
        ),
    )

    # Create poet and critic agents
    poet = client.create_agent(
        name="poet",
        instructions="You are a haiku poet. Write or revise haikus based on feedback.",
    )

    critic = client.create_agent(
        name="critic",
        instructions=(
            "You are a haiku critic. Provide 2-3 specific, actionable suggestions "
            "for improvement. Focus on imagery, syllable count, seasonal words, "
            "or emotional impact. Be constructive and brief. If satisfied with "
            "the haiku and your comments have been addressed, respond with 'APPROVED'."
        ),
    )

    # Track iteration count
    max_iterations = 4
    iteration_count = [0]  # Use list for mutability in closure

    # Create custom executor for checking approval and controlling loop
    @executor
    async def check_approval(
        response: AgentExecutorResponse,
        context: WorkflowContext[AgentExecutorRequest, str],
    ) -> None:
        """Check if critic approved or if we've hit max iterations."""
        iteration_count[0] += 1

        if response.full_conversation:
            last_message = response.full_conversation[-1]

            # Check for approval or max iterations
            if "APPROVED" in last_message.text.upper():
                await context.yield_output("Haiku approved!")
                return

            if iteration_count[0] >= max_iterations:
                await context.yield_output(
                    f"Max iterations ({max_iterations}) reached."
                )
                return

        # Continue the loop - send conversation back to poet
        await context.send_message(
            AgentExecutorRequest(
                messages=response.full_conversation,
                should_respond=True,
            )
        )

    # Build cyclic workflow: poet -> critic -> check -> (back to poet or exit)
    workflow = (
        WorkflowBuilder()
        .add_edge(poet, critic)
        .add_edge(critic, check_approval)
        .add_edge(check_approval, poet)  # Loop back
        .set_start_executor(poet)
        .build()
    )

    # Run the workflow
    task = "Write a haiku about cherry blossoms in spring"
    print(f"Task: {task}")
    print("Poet and Critic collaboration:\n")
    print("=" * 50)

    current_agent = None
    async for event in workflow.run_stream(task):
        if isinstance(event, WorkflowOutputEvent):
            print(f"\n{'=' * 50}")
            print(f"Result: {event.data}")
        elif isinstance(event, AgentRunUpdateEvent):
            # Print agent header when switching
            if current_agent != event.executor_id:
                if current_agent is not None:
                    print("\n")
                print(f"--- {event.executor_id} ---")
                current_agent = event.executor_id

            # Print streamed text
            if event.data and event.data.text:
                print(event.data.text, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
