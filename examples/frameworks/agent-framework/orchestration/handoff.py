"""
Handoff Orchestration Example - Microsoft Agent Framework

This example demonstrates the handoff pattern where a triage agent
routes requests to specialized agents. This pattern is native to
Agent Framework and shows a powerful orchestration capability.

The triage agent receives all user input first and decides which
specialist (refund, shipping, or general support) should handle it.

Run: python examples/frameworks/agent-framework/orchestration/handoff.py

Prerequisites:
    - pip install agent-framework[azure]
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    - Either set AZURE_OPENAI_API_KEY or run `az login` for Azure CLI auth
"""

import asyncio
import os
from collections.abc import AsyncIterable
from typing import cast

from agent_framework import (
    ChatMessage,
    HandoffBuilder,
    HandoffUserInputRequest,
    RequestInfoEvent,
    WorkflowEvent,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
)
from agent_framework.azure import AzureOpenAIChatClient


def create_agents(client: AzureOpenAIChatClient):
    """Create triage and specialist agents."""

    # Triage agent - routes to specialists
    triage = client.create_agent(
        name="triage_agent",
        instructions=(
            "You are frontline support triage. Read the user message and decide "
            "whether to hand off to refund_agent, shipping_agent, or support_agent. "
            "Provide a brief response and call the appropriate handoff tool "
            "(handoff_to_refund_agent, handoff_to_shipping_agent, or "
            "handoff_to_support_agent) when delegation is needed."
        ),
    )

    # Refund specialist
    refund = client.create_agent(
        name="refund_agent",
        instructions=(
            "You handle refund workflows. Ask for order identifiers if needed "
            "and outline the refund steps clearly."
        ),
    )

    # Shipping specialist
    shipping = client.create_agent(
        name="shipping_agent",
        instructions=(
            "You resolve shipping and delivery issues. Clarify the problem "
            "and describe actions to remedy it."
        ),
    )

    # General support
    support = client.create_agent(
        name="support_agent",
        instructions=(
            "You are general support. Offer empathetic troubleshooting for "
            "issues that don't match other specialists."
        ),
    )

    return triage, refund, shipping, support


async def drain_events(stream: AsyncIterable[WorkflowEvent]) -> list[WorkflowEvent]:
    """Collect all events from the stream."""
    return [event async for event in stream]


def process_events(events: list[WorkflowEvent]) -> list[RequestInfoEvent]:
    """Process events and return any pending user input requests."""
    requests: list[RequestInfoEvent] = []

    for event in events:
        if isinstance(event, WorkflowStatusEvent):
            if event.state in {
                WorkflowRunState.IDLE,
                WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
            }:
                print(f"[Status: {event.state.name}]")

        elif isinstance(event, WorkflowOutputEvent):
            conversation = cast(list[ChatMessage], event.data)
            if isinstance(conversation, list):
                print("\n=== Final Conversation ===")
                for msg in conversation:
                    speaker = msg.author_name or msg.role.value
                    print(f"  {speaker}: {msg.text[:100]}...")
                print("=" * 30)

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                print("\n--- Conversation So Far ---")
                for msg in event.data.conversation:
                    speaker = msg.author_name or msg.role.value
                    print(f"  {speaker}: {msg.text}")
                print("-" * 30)
            requests.append(event)

    return requests


async def main():
    """Demonstrate handoff orchestration with scripted responses."""
    print("=== Handoff Orchestration Example (Agent Framework) ===\n")

    # Create Azure client
    client = AzureOpenAIChatClient(
        deployment_name=os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"
        ),
    )

    # Create agents
    triage, refund, shipping, support = create_agents(client)

    # Build handoff workflow
    workflow = (
        HandoffBuilder(
            name="customer_support",
            participants=[triage, refund, shipping, support],
        )
        .set_coordinator("triage_agent")
        .with_termination_condition(
            # Stop after 3 user messages
            lambda conv: sum(1 for m in conv if m.role.value == "user") >= 3
        )
        .build()
    )

    # Scripted user responses (in production, these would be real user input)
    scripted_responses = [
        "My order #12345 hasn't arrived yet. It's been 2 weeks!",
        "Yes, please help me track it or get a refund.",
    ]

    # Start workflow
    print("Starting customer support workflow...\n")
    initial_message = "Hello, I need help with an order."

    events = await drain_events(workflow.run_stream(initial_message))
    pending = process_events(events)

    # Process request/response cycle
    while pending and scripted_responses:
        user_response = scripted_responses.pop(0)
        print(f"\n[User]: {user_response}")

        # Send response to all pending requests
        responses = {req.request_id: user_response for req in pending}
        events = await drain_events(workflow.send_responses_streaming(responses))
        pending = process_events(events)

    print("\n=== Workflow Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
